"""
Provide a simple, easy to use model for fitting data and especially
distributions
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


from future import standard_library
standard_library.install_aliases()

from functools import reduce

import numpy as np
import dashi as d
import pylab as p

from copy import deepcopy as copy

import scipy.optimize as optimize
import seaborn.apionly as sb

from . import functions as funcs

PALETTE = sb.color_palette("dark")


class Model(object):
    """
    Model data with a parametrized prediction
    """

    def __init__(self, func, startparams=None, func_norm=1):
        """
        Initialize a new model


        Args:
            func: the function to predict the data
            startparams (tuple): A set of startparameters.
            norm: multiply the result of func when evaluated with norm
        """

        # if no startparams are given, construct 
        # and initialize with 0es.
        # FIXME: better estimate?
        if startparams is None:
            startparams = [0]*(func.__code__.co_argcount - 1) # first argument is not a free parameter.

        def normed_func(*args):
            return func_norm*func(*args)

        self._callbacks = [normed_func]
        self.startparams = list(startparams)
        self.n_params = [len(startparams)]
        self.best_fit_params = list(startparams)
        self.coupling_variable = []
        self.all_coupled = False
        self.data = None
        self.xs = None
        self.chi2_ndf = None
        self.chi2_ndf_components = []
        self.norm = 1
        self.ndf = 1
        self.prediction = lambda xs: reduce(lambda x, y: x + y,\
                                  [f(xs) for f in self.components])
        self.first_guess = None
        self._distribution = None

    @property
    def distribution(self):
        return self._distribution

    def set_distribution(self, distr):
        """
        Assing a distribution to the model

        Args:
            distr:

        Returns:

        """
        self._distribution = distr

    @property
    def n_free_params(self):
        """
        The number of free parameters of this model

        Returns:
            int
        """
        return sum(self.n_free_params)

    def _create_distribution(self,data, nbins, normalize=False):
        """
        Create a distribution

        Returns:
            None
        """
        bins = np.linspace(min(data), max(data), nbins)

        h = d.factory.hist1d(data, bins)
        self.set_distribution(h)

        self.norm = 1
        if normalize:
            h_norm = h.normalized(density=True)
            norm = h.bincontent / h_norm.bincontent
            norm = norm[np.isfinite(norm)][0]
            self.norm = norm
            self.set_distribution(h_norm)

        self.data = self.distribution.bincontent
        self.xs = self.distribution.bincenters

    def add_first_guess(self, func):
        """
        Use func to estimate better startparameters

        Args:
            func: Has to yield a set of startparameters

        Returns:

        """

        assert self.all_coupled or len(self.n_params) == 1, "Does not work yet if not all variables are coupled"
        self.first_guess = func

    def eval_first_guess(self, data):
        """
        Assign a new set of start parameters obtained by calling the
        first geuss metthod

        :param data:
        :return:
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
            coupling_variable: variable number of the number in startparams

        Returns:
            None
        """
        assert len(np.unique(self.n_params)) == 1,\
            "Models have different numbers of parameters,difficult to couple!"

        self.coupling_variable.append(coupling_variable)

    def couple_all_models(self):
        """
        Use the first models startparams for
        the combined model

        Returns:
            None
        """
        self.all_coupled = True
        # if self.all_coupled:
        self.startparams = self.startparams[0:self.n_params[0]]

    def __add__(self, other):
        """
        Add another component to the model

        Args:
            other:

        Returns:

        """
        newmodel = Model(lambda x :x)

        # hack the new model to be like self
        newmodel._callbacks = self._callbacks + other._callbacks
        newmodel.startparams = self.startparams + other.startparams
        newmodel.n_params = self.n_params + other.n_params
        newmodel.best_fit_params = newmodel.startparams
        # self._callbacks = self._callbacks + other._callbacks
        # self.startparams = self.startparams + other.startparams
        # self.n_params = self.n_params + other.n_params
        # self.best_fit_params = self.startparams
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
            yield lambda xs: tmpcmp(xs, *best_fit)

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
        first = self._callbacks[0](xs, *firstparams)

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
            first += cmp(xs, *theparams)
        return first

    def add_data(self, data, nbins=200,\
                 create_distribution=False,\
                 normalize=False,\
                 xs=None,\
                 subtract=None):
        """
        Add some data to the model, in preparation for the fit


        Args:
            data (np.array):

        Keyword Args
            nbins (int):
            subtract (callable):
            normalize (bool): normalize the data before adding

        Returns:

        """
        if create_distribution:
            self._create_distribution(data, nbins, normalize)
            self.ndf = nbins - len(self.startparams)
        else:
            assert xs is not None, "Have to give xs if not histogramming!" 
            self.data = data
            self.xs = xs
            self.ndf = len(data) - len(self.startparams)
        if subtract is not None:
            self.data -= subtract(self.xs)

    def fit_to_data(self, silent=False, **kwargs):
        """
        Apply this model to data

        Args:
            data (np.ndarray): the data, unbinned
            silent (bool): silence output
            **kwargs: will be passed on to scipy.optimize.curvefit

        Returns:
            None
        """
        def model(xs, *params):
            thecomponents = []
            firstparams = params[0:self.n_params[0]]
            first = self._callbacks[0](xs, *firstparams)

            lastslice = self.n_params[0]
            for i, cmp in enumerate(self._callbacks[1:]):
                thisslice = slice(lastslice, self.n_params[1:][i] + lastslice)
                #tmpcmp = copy(cmp)
                theparams = list(params[thisslice])
                if self.coupling_variable:
                    for k in self.coupling_variable:
                        theparams[k] = firstparams[k]
                elif self.all_coupled:
                    theparams = firstparams
                #thecomponents.append(lambda xs: tmpcmp(xs, *params[thisslice]))
                lastslice += self.n_params[1:][i]
                first += cmp(xs, *theparams)
            return first

        startparams = self.startparams

        if not silent: print("Using start params...", startparams)

        fitkwargs = {"maxfev": 1000000, "xtol": 1e-10, "ftol": 1e-10}
        if "bounds" in kwargs:
            fitkwargs.pop("maxfev")
            # this is a matplotlib quirk
            fitkwargs["max_nfev"] = 1000000
        fitkwargs.update(kwargs)
        parameters, covariance_matrix = optimize.curve_fit(model, self.xs,\
                                                           self.data, p0=startparams,\
                                                           # bounds=(np.array([0, 0, 0, 0, 0] + [0]*len(start_params[5:])),\
                                                           # np.array([np.inf, np.inf, np.inf, np.inf, np.inf] +\
                                                           # [np.inf]*len(start_params[5:]))),\
                                                           # max_nfev=100000)
                                                           # method="lm",\
                                                           **fitkwargs)

        if not silent: print("Fit yielded parameters", parameters)
        if not silent: print("{:4.2f} NANs in covariance matrix".format(len(covariance_matrix[np.isnan(covariance_matrix)])))
        if not silent: print("##########################################")

        # simple GOF
        #norm = 1
        #if normalize:
        #    norm = h.bincontent / h_norm.bincontent
        #    norm = norm[np.isfinite(norm)][0]

        #self.norm = norm
        chi2 = (funcs.calculate_chi_square(self.data, self.norm * model(self.xs, *parameters)))
        self.chi2_ndf = chi2/self.ndf

        # FIXME: new feature
        #for cmp in self.components:
        #    thischi2 = (calculate_chi_square(h.bincontent, norm * cmp(h.bincenters)))
        #    self.chi2_ndf_components.append(thischi2/nbins)

        if not silent: print("Obtained chi2 and chi2/ndf of {:4.2f} {:4.2f}".format(chi2, self.chi2_ndf))
        self.best_fit_params = parameters
        return parameters
        #self.best_fit_params = fit_model(data, nbins, model, startparams, **kwargs)

    def clear(self):
        """
        Reset the model

        Returns:
            None
        """
        self.__init__(self._callbacks[0], self.startparams[:self.n_params[0]])

    def plot_result(self, ymin=1000, xmax=8, ylabel="normed bincount",\
                    xlabel="Q [C]", fig=None,\
                    log=True,\
                    model_alpha=.3,\
                    add_parameter_text=((r"$\mu_{{SPE}}$& {:4.2e}\\",0),),
                    histostyle="scatter",
                    datacolor="k",
                    modelcolor=PALETTE[2]):
        """
        Show the fit result

        Args:
            ymin (float): limit the yrange to ymin
            xmax (float): limit the xrange to xmax
            model_alpha (float): 0 <= x <= 1 the alpha value of the lineplot
                                for the model
            ylabel (str): label for yaxis
            log (bool): plot in log scale
            fig (pylab.figure): A figure instance
            add_parameter_text (tuple): Display a parameter in the table on the plot
                                        ((text, parameter_number), (text, parameter_number),...)
            datacolor (matplotlib color compatible)
            modelcolor (matplotlib color compatible)


        Returns:
            pylab.figure
        """
        assert self.chi2_ndf is not None, "Needs to be fitted first before plotting!"

        if fig is None:
            fig = p.figure()
        ax = fig.gca()
        if self.distribution is not None:
            self.distribution.__getattribute__(histostyle)(color=datacolor)
        else:
            ax.plot(self.xs, self.data, color=datacolor)

        infotext = r"\begin{tabular}{ll}"

        ax.plot(self.xs, self.prediction(self.xs), color=PALETTE[2], alpha=model_alpha)
        for comp in self.components:
            ax.plot(self.xs, comp(self.xs), linestyle=":", lw=1, color="k")

        infotext += r"$\chi^2/ndf$ & {:4.2f}\\".format(self.chi2_ndf)

        ax.set_ylim(ymin=ymin)
        ax.set_xlim(xmax=xmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

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
        if log: ax.set_yscale("log")
        sb.despine()
        return fig

