"""
Provide a simple, easy to use model for fitting data and especially
distributions. The model is capable of having "components", which can
be defined and fitted individually.
"""


from functools import reduce

import numpy as np
import dashi as d
import pylab as p
import iminuit
import types
import functools
import inspect
from copy import deepcopy as copy

import scipy.optimize as optimize
import seaborn as sb

from . import functions as funcs
from .. import utils as u
Logger = u.Logger

default_color = "r"
try:
    default_color = sb.color_palette("dark")[2]
except Exception as e:
    Logger.warn("Can not use seaborn dark colorpalette for setting the default color! Exception thrown {}".\
                format(e))


def copy_func(f):
    """
    Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)

    Basically recreate the function f independently.

    Args:
        f (callable): the function f will be cloned
    """
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g

def concat_functions(fncs):
    """
    Inspect functions and construct a new one which returns the added result.
    concat_functions(A(x, apars), B(x, bpars)) -> C(x, apars,bpars)
    C(x, apars, bpars) returns (A(x, apars) + B(x, bpars))

    Arguments:
        fncs (list):
            The callables to concat

    Returns:
        tuple (callable, list(pars))

    """

    datapar = inspect.getfullargspec(fncs[0]).args[0]
    pars = [inspect.getfullargspec(f).args[1:] for f in fncs]
    npars = [len(prs) for prs in pars]
    renamed_pars = [[k + str(i) for k in ipars] for i,ipars in enumerate(pars)]
    joint_pars = reduce( lambda x,y : x + y, renamed_pars)

    slices = []
    lastslice = 0
    for i, pr in enumerate(npars):
        thisslice = slice(lastslice, npars[i] + lastslice)
        slices.append(thisslice)
        lastslice += npars[i]
    globals().update({"concat_functions_fncs":fncs,\
                      "concat_functions_slices" : slices,\
                      "concat_functions_joint_pars" : joint_pars,\
                      "concat_functions_datapar" : datapar})
    TEMPLATE = """def jointfunc({}):
    data = concat_functions_fncs[0](*([{}] + ([{}][concat_functions_slices[0]])))
    for i,fn in enumerate(concat_functions_fncs[1:]):
        data += fn(*([{}] + ([{}][concat_functions_slices[1:][i]])))
    return data
""".format(datapar + "," + ",".join(joint_pars),datapar, ",".join(joint_pars),datapar, ",".join(joint_pars))
    exec (TEMPLATE, globals())
    return jointfunc, joint_pars

def construct_efunc(x, data, jointfunc, joint_pars):
    """
    Construct a least-squares error function. This function will then 
    be minimized, e.g. with the help of minuit.

    Args:
        x            (np.ndarray): The x-values the fit should be evaluated on
        data:        (np.ndarray): The y-values of the data we want to describe
        jointfunc:     (callable): The full data model with all components
        joint_pars:       (tuple): The model parameters   

    Returns:
        callable                 
    """
    # ensure that "x" is the first parameter, even if it 
    # has a different name
    datapar = inspect.getfullargspec(jointfunc).args[0]
    globals().update({"jointfunc" : jointfunc,\
                      "{}".format(datapar) : x,\
                      "data" : data})
    EFUNC = """def efunc({}):
    return ((abs(data - jointfunc({}))**2).sum())""".format(",".join(joint_pars), datapar + "," + ",".join(joint_pars))

    exec (EFUNC, globals())
    return efunc

def create_minuit_pardict(fn, startparams, errors, limits, errordef):
    """
    Construct a dictionary for minuit fitting. This dictionary contains information
    for the minuit fitter like startparams or limits.

    Args:
        fn        (callable): The function for which 
        startparams  (tuple): A list of startparameter. One each per parameter
        errors        (list): ?
        limits (list(tuple)): A list of (min, max) tuples for each parameter, can be None
        errordef     (float): The errordef should be 1 for a least square fit 
                              (for what this all is constructed for) or 0.5 in 
                              case of a likelihood fit
    Returns:
        dict
    """
    parnames = inspect.getfullargspec(fn).args
    mindict = dict()
    for i,k in enumerate(parnames):
        mindict[k] = startparams[i]
        if not errors is None: mindict["error_" + k] = errors[i]
        if not limits is None: mindict["limit_" + k] = (limits[i][0], limits[i][1])
    mindict["errordef"] = errordef
    return mindict

class Model(object):
    """
    Describe data with a prediction. The Model class allows to set a function
    for data prediction, and fit it to the data by the means of a chi2 fit.
    It is possible to use a collection of functions to describe a complex model,
    e.g Gaussian + some exponential tail.
    The individual models can be fitted independently, which results in sum_i n_i de
    degrees of freedom for i models with n_i parameters each, or alternatively they c
    can be coupled and share parameters, which results in sum_i n_i - n_ij degrees of
    freedom where n_ij is a shared parameters.
    """

    def __init__(self, func,\
                 startparams=None,\
                 limits=((-np.inf, np.inf),),\
                 errors=(10.,),
                 func_norm=1):
        """
        Args:
            func              (fnc): The function which shall model the data.
                                     It has to be of the form f(x, par1, par2, ...).
                                     Only 1d fits are supported, and "x" must be the
                                     first argument.
        Keyword Args:
            Will be passed to iminuit:    
                startparams (tuple): A set of startparameters. 1 start parameter
                                     per function parameter. A good choice of 
                                     start parameters helps the fit a lot.
                limits      (tuple): individual limit min/max for each parameter
                                     1 tuple (min/max) per parameter
                errors      (tuple): One value per parameter, giving an 1sigma error
                                     estimate
            Additional keywords:
                func_norm   (float): multiply the result of func when evaluated with norm
        """

        # if no startparams are given, construct 
        # and initialize with 0es.
        # FIXME: better estimate?
        if startparams is None:
            startparams = [0]*(func.__code__.co_argcount - 1) # first argument is not a free parameter.

        #def normed_func(*args):
        #    return func_norm*func(*args)

        self._callbacks = [copy_func(func)]
        self.startparams = list(startparams)
        #self.errors = [len(startparams)]
        self.errors = None
        self.n_params = [len(startparams)]
        self.best_fit_params = list(startparams)
        self.coupling_variable = []
        self.all_coupled = False
        self.data = None
        self.data_errs = None
        self.xs = None
        self.chi2_ndf = None
        self.chi2_ndf_components = []
        self.norm = 1
        self.ndf = 1
        self.func_norm = [float(func_norm)]
        self.prediction = lambda xs: reduce(lambda x, y: x + y,\
                                  [self.func_norm[i]*f(xs) for i,f in enumerate(self.components)])
        self.first_guess = None
        self._distribution = None
        self._is_categorical = False 
        self.covariance_matrix = None

    @property
    def distribution(self):
        return self._distribution

    def set_distribution(self, distr):
        """
        Adding a distribution to the model. The distribution shall contain
        the data we want to model.

        Args:
            distr (dashi.histogram):

        Returns:
            None
        """
        self._distribution = distr

    #def construct_minuit_function(self):
    #    func_args, coupling = self.extract_parameters()
    #    parameter_set = ["a","b","c","d","e","f","g"]
    #    if self.all_coupled:
    #        func_args = func_args[0]
    #        # x is always 0
    #        fnc_src = """def to_minimize({}):\n\t
    #        """
    #        #def to_minimize()


    @property
    def n_free_params(self):
        """
        The number of free parameters of this model. The free 
        parameter in a least square fit are 
        number of data points - fit parameters.

        Returns:
            int
        """
        return sum(self.n_free_params)

    def _create_distribution(self, data, bins,\
                             normalize=False, density=True):
        """
        Create a distribution out of the data with the help of 
        dashi.

        Args:
            data        (np.ndarray): input data
            bins (np.ndarray or int): bins to forward to the histogram creation

        Keyword Args:
            normalize         (bool): normalize the resulting distribution by nentries
            density           (bool): if together with normalize, then normalize the 
                                      distribution by area, as for a pdf for example.
                                      The area under the curve will then be unity.
        Returns:
            None
        """
        if np.isscalar(bins):
            bins = np.linspace(min(data), max(data), bins)

        h = d.factory.hist1d(data, bins)

        if normalize:
            h_norm = h.normalized(density=density)
            norm = h.bincontent / h_norm.bincontent
            norm = norm[np.isfinite(norm)][0]
            self.norm = norm
            self.set_distribution(h_norm)
        else:
            self.norm = 1
            self.set_distribution(h)
        self.data = self.distribution.bincontent
        self.xs = self.distribution.bincenters

    def add_first_guess(self, func):
        """
        Use func to estimate better startparameters for the initialization 
        of the fit.

        Args:
            func (callable): The function func has to have the same amount
                             of parameters as we have startparameters.

        Returns:
            None
        """

        assert self.all_coupled or len(self.n_params) == 1, "Does not work yet if not all variables are coupled"
        self.first_guess = func

    def eval_first_guess(self, data):
        """
        Assign a new set of start parameters obtained by calling the
        first geuss metthod

        Args:
            data (np.ndarray): input data, used to evaluate the first guess method.
        Returns:
            None
        """
        assert self.first_guess is not None, "No first guess method provided! Run Model.add_first_guess "
        newstartparams = list(self.first_guess(data))
        assert len(newstartparams) == len(self.startparams), "first guess algorithm must yield startparams!"
        self.startparams = newstartparams

    def couple_models(self, coupling_variable):
        """
        Couple the models by a variable, which means use the variable not
        independently in all model components, but fit it only once.
        E.g. if there are 3 models with parameters p1, p2, k each and they
        are coupled by k, parameters p11, p21, p12, p22, and k will be fitted
        instead of p11, p12, k1, p21, p22, k2.

        Args:
            coupling_variable: variable number of the number in startparams. 
                               This *must* be the index to the respective tuple.

        Returns:
            None
        """
        assert len(np.unique(self.n_params)) == 1,\
            "Models have different numbers of parameters,difficult to couple!"

        self.coupling_variable.append(coupling_variable)

    def construct_error_function(self, startparams, errors, limits, errordef):
        """
        Construct the error function together with the necessary parameters 
        for minuit.
        
        Args:
            startparams (tuple): A set of startparameters. 1 start parameter
                                 per function parameter. A good choice of 
                                 start parameters helps the fit a lot.
            limits      (tuple): individual limit min/max for each parameter
                                 1 tuple (min/max) per parameter
            errors      (tuple): One value per parameter, giving an 1sigma error
                                 estimate
            errordef    (float): The errordef should be 1 for a least square fit 
                                 (for what this all is constructed for) or 0.5 in 
                                 case of a likelihood fit
        Returns:
            tuple (callable, dict)
        """

        concated, concated_pars = concat_functions(self._callbacks)
        error_func = construct_efunc(self.xs, self.data, concated, concated_pars)
        pardict = create_minuit_pardict(error_func, startparams, errors, limits, errordef)
        return error_func, pardict

    def couple_all_models(self):
        """
        "Lock" the model after all components have been added. This 
        will determiine a set of startparameters.
        After this, no other models can be coupled/added any more.

        Returns:
            None
        """
        self.all_coupled = True
        # if self.all_coupled:
        self.startparams = self.startparams[0:self.n_params[0]]

    def __add__(self, other):
        """
        Add another component to the model. This allows for more
        complex functions like gaussian + exponential.

        Args:
            other (Model): A full flexed model which has to be 
                           initialized with its own set of 
                           startparameters

        Returns:
            Model        : A multi-compoment Model
        """
        newmodel = Model(lambda x :x)

        # hack the new model to be like self
        newmodel._callbacks = self._callbacks + other._callbacks
        newmodel.startparams = self.startparams + other.startparams
        newmodel.n_params = self.n_params + other.n_params
        newmodel.best_fit_params = newmodel.startparams
        newmodel.func_norm = self.func_norm + other.func_norm
        return newmodel

    @property
    def components(self):
        lastslice = 0
        thecomponents = []
        for i, cmp in enumerate(self._callbacks):
            thisslice = slice(lastslice, self.n_params[i] + lastslice)
            tmpcmp = copy(cmp) # hack - otherwise it will not work :\
            lastslice += self.n_params[i]
            best_fit = self.best_fit_params[thisslice]
            if self.all_coupled:
                best_fit = self.best_fit_params[0:self.n_params[0]]
            yield lambda xs: self.func_norm[i]*tmpcmp(xs, *best_fit)

    def extract_parameters(self):
        """
        Get the variable names and coupling references for
        the individual model components

        Returns:
            tuple
        """
        all_pars = []
        for i in self._callbacks:
            all_pars.append(iminuit.describe(i))
        return all_pars, self.coupling_variable

    def __call__(self, xs, *params):
        """
        Return the model prediction

        Args:
            xs (np.ndaarray): the values the model should be evaluated on

        Returns:
            np.ndarray
        """
        thecomponents = []
        firstparams = params[0:self.n_params[0]]
        first = self.func_norm[0]*self._callbacks[0](xs, *firstparams)

        lastslice = self.n_params[0]
        for i, cmp in enumerate(self._callbacks[1:]):
            thisslice = slice(lastslice, self.n_params[1:][i] + lastslice)
            # tmpcmp = copy(cmp)
            theparams = list(params[thisslice])
            if self.coupling_variable:
                for k in self.coupling_variable:
                    theparams[k] = firstparams[k]
            elif self.all_coupled:
                theparams = firstparams
            # thecomponents.append(lambda xs: tmpcmp(xs, *params[thisslice]))
            lastslice += self.n_params[1:][i]
            first += self.func_norm[i]*cmp(xs, *theparams)
        return first

    def add_data(self, data,\
                 data_errs=None,\
                 bins=200,\
                 create_distribution=False,\
                 normalize=False,\
                 density=True,\
                 xs=None,\
                 subtract=None):
        """
        Add some data to the model, in preparation for the fit. There are two
        modes of this:
        1) Data needs to be histogrammed, then make sure to set
           'nbins' appropriatly and set the 'create_distribution'
        2) Data needs NOT to be histogrammed. In that case, bins has no meaning
           For a meaningful calculation of chi2, the errors of the data points
           need to be given to data_errs


        Args:
           data (np.array)            : input data

        Keyword Args
           data_errs (np.array)       : errors of the data for chi2 calculation
                                         (only used when not histogramming)
           nbins (int/np.array)       : number of bins or bin array to be passed 
                                         to the histogramming routine
           create_distribution (bool) : data requires the creation of a histogram
                                         first before fitting
           subtract (callable)        : ?
           normalize (bool)           : normalize the data before adding
           density (bool)             : if normalized, assume the data is a pdf.
                                        if False, use bincount for normalization.
        Returns:
           None
        """
        if np.isscalar(bins):
            nbins = bins
        else:
            nbins = len(bins)
        
        if create_distribution:
            self._create_distribution(data, bins, normalize, density=density)
            self._is_categorical = True
            self.ndf = nbins - len(self.startparams)
        else:
            assert xs is not None, "Have to give xs if not histogramming!" 
            if data_errs is not None:
                assert len(data_errs) == len(data), "Data and associated errrors must have the same size"
            else:
                data_errs = np.ones(len(data))
            self.data = data
            self.data_errs = data_errs
            self.xs = xs
            self.ndf = len(data) - len(self.startparams)
            self.bins = None
        if subtract is not None:
            self.data -= subtract(self.xs)

    def fit_to_data(self, silent=False,\
                    use_minuit=True,\
                    errors=None,\
                    limits=None,\
                    errordef=1,\
                    debug_minuit=False,\
                    **kwargs):
        """
        Apply this model to data. This will perform the fit with the help 
        of either minuit or scipy.optimize.

        Args:
            data (np.ndarray)      : the data, unbinned
            silent (bool)          : silence output
            use_minuit (bool)      : use minuit for fitting
            errors (list)          : errors for minuit, see miniuit manual
            limits (list of tuples): limits for minuit, see minuit manual
            errordef (float)       : typically 1 for chi2 fit and 0.5 for llh fit 
                                   : this class is currently set up as a leeast square
                                     fit, so this should not be changed
            debug_minuit (int)     : if True, attache the iminuit instance to the model
                                     so that it can be inspected later on. Will raise error
                                     if use_minuit is set to False at the same time
            **kwargs: will be passed on to scipy.optimize.curvefit

        Returns:
            None
        """
        if not use_minuit and debug_minuit:
            raise ValueError("You can not debug  minuit when you are not using it!")

        startparams = self.startparams

        if not silent: print("Using start params...", startparams)

        fitkwargs = {"maxfev": 1000000, "xtol": 1e-10, "ftol": 1e-10}
        if "bounds" in kwargs:
            fitkwargs.pop("maxfev")
            # this is a matplotlib quirk
            fitkwargs["max_nfev"] = 1000000
        fitkwargs.update(kwargs)
        if use_minuit:
            errorfunc, params = self.construct_error_function(self.startparams,\
                                                               errors,\
                                                               limits,\
                                                               errordef)
            # for debugging reasons we attach the minuit instance
            # to the model if desired
            m = iminuit.Minuit(errorfunc, **params)
            m.migrad()
            values = m.values
            if not silent: print (values, "result")
            parameters=[]
            for k in sorted(m.var2pos, key=m.var2pos.get):
                if not silent : print (k)
                parameters.append(m.values[k])
            self.errors = m.errors
            self.covariance_matrix = m.covariance
        else:
            parameters, covariance_matrix = optimize.curve_fit(self, self.xs,\
                                                           self.data, p0=startparams,\
                                                           # bounds=(np.array([0, 0, 0, 0, 0] + [0]*len(start_params[5:])),\
                                                           # np.array([np.inf, np.inf, np.inf, np.inf, np.inf] +\
                                                           # [np.inf]*len(start_params[5:]))),\
                                                           # max_nfev=100000)
                                                           # method="lm",\
                                                           **fitkwargs)
            self.covariance_matrix = covariance_matrix
            self.errors = [] 
            for i, row in enumerate(self.covariance_matrix):
                for j,entry in enumerate(row):
                    if i == j:
                        self.errors.append(np.sqrt(entry))
        if not silent: print("Fit yielded parameters", parameters)
        if (not silent) and (not use_minuit): print("{:4.2f} NANs in covariance matrix".format(len(self.covariance_matrix[np.isnan(np.asarray(covariance_matrix))])))

        # simple GOF
        #norm = 1
        #if normalize:
        #    norm = h.bincontent / h_norm.bincontent
        #    norm = norm[np.isfinite(norm)][0]

        #self.norm = norm
        if self._is_categorical:
            chi2 = (funcs.calculate_chi_square(self.norm*self.data, self.norm * self(self.xs, *parameters)))
        else:
            # calculate the chi2 
            chi2 = (funcs.calculate_reduced_chi_square(self.norm*self.data, self.norm * self(self.xs, *parameters), self.data_errs))
        self.chi2_ndf = chi2/self.ndf

        # FIXME: new feature
        #for cmp in self.components:
        #    thischi2 = (calculate_chi_square(h.bincontent, norm * cmp(h.bincenters)))
        #    self.chi2_ndf_components.append(thischi2/nbins)
        if not silent and use_minuit: print("Function value at minimum {:4.2e}".format(m.fval))
        if not silent: print("Obtained chi2 : {:4.2f}; ndf : {:4.2f}; chi2/ndf {:4.2f}".format(chi2, self.ndf, self.chi2_ndf))
        if not silent: print("##########################################")
        self.best_fit_params = parameters
        if debug_minuit:
            self._m_instance = m
        return parameters
    
    def get_minuit_instance(self):
        """
        If a previous fit has been done with the debug_minuit instance
        then it now can be accessed.

        """
        if not hasattr(self, '_m_instance'):
            Logger.warn('Minuit instance not available. Execute `fit_to_data` with `debug_minuit` set to True')
            return None
        else:
            return self._m_instance


    def clear(self):
        """
        Reset the model. This bascially deletes all components and 
        resets the startparameters.

        Returns:
            None
        """
        self.__init__(self._callbacks[0], self.startparams[:self.n_params[0]])

    def plot_result(self, ymin=1000, xmax=8,\
                    ylabel="normed bincount",\
                    xlabel="Q [C]", fig=None,\
                    log=True,\
                    figure_factory=None,\
                    axes_range="auto",
                    model_alpha=.3,\
                    add_parameter_text=((r"$\mu_{{SPE}}$& {:4.2e}\\",0),),
                    histostyle="scatter",
                    datacolor="k",
                    modelcolor=default_color):
        """
        Show the fit result, together with the fitted data.

        Args:
            ymin               (float): limit the yrange to ymin
            xmax               (float): limit the xrange to xmax
            model_alpha        (float): 0 <= x <= 1 the alpha value of the lineplot
                                        for the model
            ylabel               (str): label for yaxis
            log                 (bool): plot in log scale
            figure_factory       (fnc): Use to generate the figure
            axes_range           (str): the "field of view" to show
            fig         (pylab.figure): A figure instance
            add_parameter_text (tuple): Display a parameter in the table on the plot
                                        ((text, parameter_number), (text, parameter_number),...)
            datacolor            (str): color for the data points
            modelcolor           (str): color for the model prediction

        Returns:
            pylab.figure
        """
        assert self.chi2_ndf is not None, "Needs to be fitted first before plotting!"

        def auto_adjust_limits(ax, data, xs):
            scalemax, scalemin = 1.1, 0.9

            ymax, ymin = scalemax*max(data), scalemin*min(data)
            xmax, xmin = scalemax*max(xs[abs(data) > 0]), scalemin*min(xs[abs(data) >0])
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            return ax

        if figure_factory is not None:
            fig = figure_factory()

        elif fig is None:
            fig = p.figure()

        ax = fig.gca()


        if self.distribution is not None:
            self.distribution.__getattribute__(histostyle)(color=datacolor)
        else:
            ax.plot(self.xs, self.data, color=datacolor)

        infotext = r"\begin{tabular}{ll}"

        ax.plot(self.xs, self.prediction(self.xs), color=default_color, alpha=model_alpha)
        for comp in self.components:
            ax.plot(self.xs, comp(self.xs), linestyle=":", lw=1, color="k")

        infotext += r"$\chi^2/ndf$ & {:4.2f}\\".format(self.chi2_ndf)

        ax.set_ylim(ymin=ymin)
        ax.set_xlim(xmax=xmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if axes_range == "auto":
            ax = auto_adjust_limits(ax, self.data, self.xs)

        if self.distribution is not None:
            infotext += r"entries& {}\\".format(self.distribution.stats.nentries)
        if add_parameter_text is not None:
            for partext in add_parameter_text:
                infotext += partext[0].format(self.best_fit_params[partext[1]])
            #infotext += r"$\mu_{{SPE}}$& {:4.2e}\\".format(self.best_fit_params[mu_spe_is_par])
        infotext += r"\end{tabular}"
        ax.text(0.9, 0.9, infotext,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
        if log: ax.set_yscale("symlog", linthreshy=1)
        #sb.despine()
        return fig

