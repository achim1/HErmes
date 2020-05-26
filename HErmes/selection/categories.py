"""
Categories of data, like "signal" of "background" etc
"""


import pandas as pd
import inspect
import numpy as np
import dashi as d
import pylab as p
import abc
import concurrent.futures as fut
import tables
import matplotlib.colors as colors

from hepbasestack.colors import get_color_palette
from hepbasestack import isnotebook

from copy import deepcopy

from ..utils.files import harvest_files,DS_ID,EXP_RUN_ID
from ..utils import Logger

from .magic_keywords import MC_P_EN,\
                            MC_P_TY,\
                            MC_P_ZE,\
                            MC_P_WE,\
                            MC_P_GW,\
                            MC_P_TS,\
                            RUN_START,\
                            RUN_STOP, \
                            RUN,\
                            EVENT,\
                            DATASETS
from . import variables
MAX_CORES = 6

def cut_with_nans(data, cutmask):
    """
    Cut the individual fields of a 2d array and keep the 
    shape by filling up with nans

    Args:
        data (np.ndarray): The array to cut
        cutmask (np.ndarray): Cut with this boolean array

    Returns:
        np.ndarray: data with applied cuts
    """
    dlen = len(data)
    tmpdata = data[cutmask]
    tmpdata = np.hstack((tmpdata, np.nan*np.ones(dlen-len(tmpdata))))
    return tmpdata


class AbstractBaseCategory(metaclass=abc.ABCMeta):
    """
    Stands for a specific type of data, e.g.
    detector data in a specific configuarion,
    simulated data etc.
    """
    weightvarname = None
    _harvested = False

    def __init__(self, name):
        self.name = name
        self.files = None
        self.datasets = dict()
        self.vardict = dict()
        self.plot_options = dict()
        self._weightfunction = lambda x : x
        self.cuts = []
        self.cutmask = np.array([])
        self.plot = True
        self.show_in_table = True
        self._weights = pd.Series(dtype=np.float64) #dtype to suppress warning

    def __repr__(self):
        return """<{0}: {1}>""".format(self.__class__, self.name)

    def __hash__(self):
        return hash((self.name,"".join([k for k in map(str,list(self.datasets.keys()))])))

    def __eq__(self,other):
        if (self.name == other.name) and (list(self.datasets.keys()) == list(other.datasets.keys())):
            return True
        else:
            return False

    def __getitem__(self, item):
        """
        Shortcut for the get method

        Args:
            item (str):

        Returns:
            np.ndarray
        """
        return self.get(item)

    def __len__(self):
        """
        Return the longest variable element. If a cut is applied, it returns
        the *current* length of the variable data (after the cut!)
        FIXME: introduce check?
        """
        #lengths = np.array([len(self.vardict[v].data) for v in list(self.vardict.keys())])
        #lengths = np.array([len(self.get(v)) for v in list(self.vardict.keys())])
        #lengths = lengths[lengths > 0]
        #selflen = list(set(lengths))
        #if not selflen: # empty category should have len 0
        #    return 0
        #FIXME: deal with pandas DataFrames more reliably
        #FIXME: HACK
        debug = []

        for v in list(self.vardict.keys()):
            # omit parameters
            if self.vardict[v].role == variables.VariableRole.PARAMETER:
                Logger.debug("Omitting parameter {} from length calculation".format(v))
                continue 

            # if there is a cut applied, this is the
            # cutted array
            variable = self.get(v)
            if not hasattr(variable, "shape"):
                variable = np.asarray(variable)

            #print v, variable.shape
            if len(variable.shape) == 2:
                vlen = variable.shape[0]
            else:
                vlen = len(variable)
            debug.append((v, vlen))

        #debug = [(v,len(self.get(v))) for v in list(self.vardict.keys())]
        lengths = np.array([l for v,l in debug])
        lengths = lengths[lengths > 0]
        #print lengths
        selflen = list(set(lengths))
        if not selflen: # empty category should have len 0
            return 0

        assert len(selflen) == 1, "Different variable lengths for {}! {}".format(self.name,debug)
        return selflen[0]

    def distribution2d(self, varnames,
                       bins=None,
                       figure_factory=None,
                       fig=None,
                       norm=False,
                       log=True,
                       cmap=p.get_cmap("Blues"),
                       interpolation="gaussian",
                       cblabel="events",
                       weights=None,
                       transform=(None,None),
                       despine=False,
                       alpha=0.95, 
                       return_histo=False):
        """
        Draw a 2d distribution of 2 variables in the same category.
        Args:
            varnames (tuple(str,str)): The names of the variable in the catagory

        Keyword Args:
            bins (tuple(int/np.ndarray)): Bins for the distribution
            cmap : A colormap
            // alpha (float): 0-1 alpha value for histogram
            fig (matplotlib.figure.Figure): Canvas for plotting, if None an empty one will be created
            // xlabel (str): xlabel for the plot. If None, default is used
            norm (str) : "n" or "density" - make normed histogram
            // style (str): Either "line" or "scatter"
            transform (callable): Apply transformation to the data before plotting
            alpha (float) : 0-1, transparency of the histogram
            log (bool): Plot yaxis in log scale
            transform (tuple): Two functions which shall transform sample 1 and 2 respectively
            figure_factory (func): Must return a single matplotlib.Figure, NOTE: figure_factory has priority over fig keyword
            return_histo (bool): Return the histogram instead of the figure. WARNING: changes return type!
        Returns:
            matplotlib.figure.Figure or dashi.histogram.hist1d

        """
        sample = []

        for var_k,varname in enumerate(varnames):
            var = self.get(varname)
            if not isinstance(var, np.ndarray):
                var = var.as_matrix()
            if transform[var_k] is not None:
                var = transform[var_k](var)
            sample.append(var)
            

#            # XXX HACK
#            # FIXME: This doesn't really work...
#            if not hasattr(var, "ndim"):
#                var = np.asarray(var)
#            if var.ndim != 1:
#                Logger.warning("Unable to histogram array-data. Needs to be flattened (e.g. by averaging first!\
#                                Data shape is {}".format(self.get(varname).shape))
#                return fig
#
#            # not sure why this is necessary - maybe for 2d arrays_
#            # FIXME
#            if transform[var_k] is not None:
#                sample.append(np.asarray(transform[var_k](np.asarray(var))))
#            else:
#                sample.append(np.asarray(var))

        sample = tuple(sample)
        if figure_factory is not None:
            fig = figure_factory()

        if fig is None:
            fig = p.figure()

        if bins is None:
            bins = self.vardict[varnames[0]].bins, self.vardict[varnames[1]].bins

        xlabel, ylabel= None, None
        if xlabel is None:
            xlabel = self.vardict[varnames[0]].label
        if ylabel is None:
            ylabel = self.vardict[varnames[1]].label

        ax = fig.gca()
        h2 = d.factory.hist2d(sample, bins, weights=weights)
        cmap.set_bad('w', 1)
        if not norm:
            #h2.imshow(log=log, cmap=cmap, interpolation=interpolation, alpha=0.95, label='events')
            norm = None
        else:
            h2 = h2.normalized()
            minval, maxval = min(h2.bincontent.flatten()), max(h2.bincontent.flatten())
            #norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=minval, vmax = maxval)
            norm=None
        try:
            h2.imshow(log=log, cmap=cmap,
                      interpolation=interpolation, alpha=alpha, label='events',
                      norm=norm)
            cb = p.colorbar(drawedges=False, shrink=0.5, orientation='vertical', fraction=0.05)
            cb.set_label(cblabel)
            cb.ticklocation = 'left'
        except Exception as e:
            Logger.warning("Exception encountered when creating colorbar! {}".format(e))
            #Logger.warning("Creation of colorbar failed with min {:4.2e} and max {:4.2e}".format(minval, maxval))
            Logger.warning("Will try again without using normalization for colorbar...")
            #norm=colors.Normalize(minval, maxval)
            h2.imshow(log=log, cmap=cmap, interpolation=interpolation,
                  alpha=0.95, label='events')
            cb = p.colorbar(drawedges=False, shrink=0.5, orientation='vertical', fraction=0.05)
            cb.set_label(cblabel)
            cb.ticklocation = 'left'

    
        ax = fig.gca()
        ax.grid(1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if despine:
            sb.despine()
        else:
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
        if return_histo:
            return h2
        return fig

    def distribution(self, varname, bins=None,\
                     color=None, alpha=0.5,\
                     fig=None, xlabel=None,\
                     norm=False,\
                     filled=None,\
                     legend=True,\
                     style="line", log=False,
                     transform=None,
                     extra_weights=None,
                     figure_factory=None,
                     return_histo=False):
        """
        Plot the distribution of variable in the category

        Args:
            varname (str): The name of the variable in the catagory

        Keyword Args:
            bins (int/np.ndarray): Bins for the distribution
            color (str/int): A color identifier, either number 0-5 or matplotlib compatible
            alpha (float): 0-1 alpha value for histogram
            fig (matplotlib.figure.Figure): Canvas for plotting, if None an empty one will be created
            xlabel (str): xlabel for the plot. If None, default is used
            norm (str) : "n" or "density" - make normed histogram
            style (str): Either "line" or "scatter"
            filled (bool): Draw filled histogram
            legend (bool): if available, plot a legend
            transform (callable): Apply transformation to the data before plotting
            log (bool): Plot yaxis in log scale
            extra_weights (numpy.ndarray): Use this for weighting. Will overwrite any other weights in the dataset
            figure_factory (func): Must return a single matplotlib.Figure, NOTE: figure_factory has priority over fig keyword
            return_histo (bool): Return the histogram instead of the figure. WARNING: changes return type!
        Returns:
            matplotlib.figure.Figure or dashi.histogram.hist1d

        """

        if figure_factory is not None:
            fig = figure_factory()

        if fig is None:
            fig = p.figure()

        if self.get(varname).ndim != 1:
            Logger.warning("Unable to histogram array-data. Needs to be flattened (e.g. by averaging first!\
                            Data shape is {}".format(self.get(varname).shape))
            return fig

        ax = fig.gca()
        if bins is None:
            bins = self.vardict[varname].bins
        if bins is None:
            try:
                if len(self.cutmask) != 0:
                    bins = self.varname[varname].calculate_fd_bins(cutmask=self.cutmaks)   
                else:
                    bins = self.vardict[varname].calculate_fd_bins()
            except Exception as e:
                Logger.warning(f"Can not create Friedman Draconis bins {e}")
                Logger.warning("Will return 40 as last resort... Recommended to specify bins via the function parameter")
                bins = 40
        palette = get_color_palette()
        plotdict = False
        if color is None:
            color=palette[0]
        if "linestyle" in self.plot_options:
            plotdict = deepcopy(self.plot_options["linestyle"])
        if "scatterstyle" in self.plot_options:
            plotdict = deepcopy(self.plot_options["scatterstyle"])
        if "label" in self.plot_options:
            plotdict["label"] = deepcopy(self.plot_options["label"])
        if not plotdict and ((style == "line") or (style is None)):
            plotdict = {"color" : color,
                        "filled": True,
                        "alpha" : 1,
                        "linewidth": 3,
                        "linestyle": "solid",
                        "fc"    : color} 
        if not plotdict and (style == "scatter"):
            plotdict = {"color": "k",
                        "linewidth": 3,
                        "alpha": 1,
                        "marker": "o",
                        "markersize": 4}
        plotdict["color"] = palette[plotdict["color"]]
        if xlabel is None:
            xlabel = self.vardict[varname].label
        if (xlabel is None) or (not xlabel):
            xlabel = varname
        if transform is not None:
            data = transform(self.get(varname))
        else:
            data = self.get(varname)

        # FIXME weights!!
        h = d.factory.hist1d(data, bins, weights=extra_weights)
        if norm:
            #assert ((norm == "n" or norm == "density"), "Horm has to be either n or denstiy")
            if norm == "density":
                h = h.normalized(density=True)
            else:
                h = h.normalized()
        if style is None:
            try:
                style = self.plot_options["histotype"]
            except KeyError:
                Logger.warning("Can not derive plot style. Falling back to line plot!")
                style = "line"

        if style == "line":
            # FIXME always fill histos for now
            if filled is not None:
                plotdict["filled"] = filled
            plotdict["alpha"] = alpha
            h.line(**plotdict)
            if "filled" in plotdict:
                if plotdict["filled"]:
                    plotdict["filled"] = False
                    plotdict["label"] = "_nolegend_"
                    h.line(**plotdict)
    
            #h.line(filled=True,
            #       color=color,
            #       fc=color,
            #       alpha=alpha)  # hatch="//")
            #h.line(color=color)
        elif style == "scatter":
            h.scatter(**plotdict)

        else:
            raise ValueError("Can not understand style {}. Has to be either 'line' or 'scatter'".format(style))
        ax.set_ylabel("events")
        ax.set_xlabel(xlabel)
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=min(h.binedges))
        if log:
            #ax.semilogy(nonposy="clip")
            ax.set_yscale("symlog")
        if ("label" in plotdict) and legend:
            ax.legend()
        if return_histo:
            return h
        return fig

############################################################################

    @property
    def raw_count(self):
        """
        Gives a number of "how many events are actually there"

        Returns:
            int
        """
        assert self.harvested,"Please read out variables first"
        return len(self)

    @property
    def variablenames(self):
        return list(self.vardict.keys())

    @property
    def harvested(self):
        return self._harvested

    def declare_harvested(self):
        """
        Set the flag that all the variables have been read out
        """
        self._harvested = True

    #def set_weightfunction(self,func):
    #    """
    #    Register a function used for weighting
    #
    #    Args:
    #        func (func): the function to be used
    #    """
    #    self._weightfunction = func

    def add_cut(self,cut):
        """
        Add a cut without applying it yet

        Args:
            cut (pyevsel.variables.cut.Cut): Append this cut to the internal cutlist

        """

        self.cuts.append(cut)

    def apply_cuts(self, inplace=False):
        """
        Apply the added cuts.

        Keyword Args:
            inplace (bool): If True, cut the internal variable buffer
                           (Can not be undone except variable is reloaded)
        """
        self.undo_cuts()
        mask = np.ones(self.raw_count)
        if self.cuts[0].type == 'OR':
            Logger.warning("'OR' type cut is experimental!")
            mask = np.zeros(self.raw_count)

        # only apply the condition to the mask
        # created for the cut with the condition
        # not the others
        cond_masks = []
        for cut in self.cuts:
            cond_mask = np.ones(self.raw_count)
            if cut.condition is None:
                continue

            for varname,(op,value) in cut:
                s = self.get(varname)
                cond_mask = np.logical_and(cond_mask, op(s,value) )
            cond_mask = np.logical_or(cond_mask, np.logical_not(cut.condition[self.name]))
            cond_masks.append(cond_mask)

        # finish the conditional part
        for m in cond_masks:
            mask = np.logical_and(mask,m)

        # only for non-conditional cuts
        for cut in self.cuts:
            if cut.condition is not None:
                continue

            thiscutmask = np.ones(self.raw_count)
            if cut.type == 'OR':
                Logger.warning("'OR' type cut is experimental!")
                thiscutmask = np.zeros(self.raw_count)
            
            for varname, (op,value) in cut:
                s = self.get(varname)
                Logger.debug("Cutting on {}".format(varname))
                # special treatement if variable
                # is an array
                if (self[varname].ndim == 2) or (len(self[varname].shape) == 2):
                    Logger.warning("Cut on array variable {} can be only applied inline!".format(varname))
                    Logger.warning("Conditions can not be applied to array variable!")
                    for i, k in enumerate(s):
                        tmpmask =  op(s[i],value) 
                        s[i] = cut_with_nans(s[i], tmpmask)
                    self.vardict[varname]._data = s
                # in the case of a jagged array, it will be recognized as 
                # one dimensional.
                # However, the entries of the array are iterables
                elif hasattr(self.vardict[varname]._data[0],"__iter__"):
                    #Logger.warning("Cut on jagged array for variable {}! Can only be applied inplace!".format(varname))
                    Logger.warning("Conditions can not be applied to array variable! However, this will happen in a future version...")
                    Logger.warning("Cut on jagged array for varialbe {} is currently an experimental feature".format(varname))
                    #for i, k in enumerate(s):
                    #    tmpmask =  op(s[i],value) 
                    #    s[i] = cut_with_nans(s[i], tmpmask)
                    #self.vardict[varname]._data = s
                    #inplace = True        
                    assert len(mask) == len(op(s, value)),\
                        "Cutting fails due to different variable lengths for {}".format(varname)
                    if cut.type == 'OR':
                        Logger.warning("'OR' cut is still experimental!")
                        thiscutmask = np.logical_or(thiscutmask, op(s, value))
                    else:
                        thiscutmask = np.logical_and(thiscutmask, op(s, value))
                else:
                    assert len(mask) == len(op(s, value)),\
                        "Cutting fails due to different variable lengths for {}".format(varname)
                    if cut.type == 'OR':
                        Logger.warning("'OR' cut is still experimental!")
                        thiscutmask = np.logical_or(thiscutmask, op(s, value))
                    else:
                        thiscutmask = np.logical_and(thiscutmask, op(s,value))
            if not mask.any():
                mask = np.logical_or(thiscutmask, mask)
            else:
                mask = np.logical_and(thiscutmask, mask)
        if inplace:
            for k in list(self.vardict.keys()):
                if self.vardict[k].ndim != 2:
                    self.vardict[k]._data = self.vardict[k].data[mask]
        else:
            # multidim variables are cut inline anyway
            self.cutmask = np.array(mask, dtype=bool)
        #elif self.vardict[varname].ndim != 2:
        #    self.cutmask = np.array(mask, dtype=bool)
        #else:
        #    pass

    def undo_cuts(self):
        """
        Conveniently undo a previous "apply_cuts"
        """

        self.cutmask = np.array([])

    def delete_cuts(self):
        """
        Get rid of previously added cuts and undo them
        """

        self.undo_cuts()
        self.cuts = []

    @staticmethod
    def _ds_regexp(filename):
        """
        A container for matching a dataset number against a filename

        Args:
            filename (str): An filename of a datafile
        Returns:
            dataset (int): A dataset number extracted from the filename
        """
        return DS_ID(filename)

    def load_vardefs(self, module):
        """
        Load the variable definitions from a module

        Args:
            module (python module): Needs to contain variable definitions
        """

        all_vars = inspect.getmembers(module)
        all_vars = [x[1] for x in all_vars if isinstance(x[1], variables.AbstractBaseVariable)]
        Logger.info("Found {} variables".format(len(all_vars)))
        for v in all_vars:
            if v.name in self.vardict:
                Logger.warning("Variable {} already defined,skipping!".format(v.name))
                continue
            self.add_variable(v)

    def add_variable(self,variable):
        """
        Add a variable to this category

        Args:
            variable (pyevsel.variables.variables.Variable): A Variable instalce
        """
        tmpvar = deepcopy(variable)
        self.vardict[tmpvar.name] = tmpvar

    def drop_empty_variables(self):
        """
        Delete variables which have no len

        Returns:
            None
        """
        to_delete = []
        for k in self.vardict.keys():
            if not len(self.vardict[k].data):
                to_delete.append((k))

        for k in to_delete:
            Logger.info("Deleting empty variable {}".format(k))
            self.delete_variable(k)


    def delete_variable(self, varname):
        """
        Remove a variable entirely from the category

        Args:
            varname (str): The name of the variable as stored in self.variable dict

        Returns:
            None
        """
        if not varname in self.vardict:
            Logger.warning("Can not delete variable {} for category {}!".format(varname, self.name))
            return
        delvar = self.vardict.pop(varname)
        del delvar


    def get_files(self, *args, **kwargs):
        """
        Load files for this category
        uses HErmes.utils.files.harvest_files

        Args:
            *args (list of strings): Path to possible files

        Keyword Args:
            datasets (dict(dataset_id : nfiles)): i given, load only files from dataset dataset_id  set nfiles parameter to amount of L2 files the loaded files will represent
            force (bool): forcibly reload filelist (pre-readout vars will be lost)
            append (bool): keep the already aquired files and only append the new ones
            all other kwargs will be passed to
            utils.files.harvest_files

        """

        force = False
        append = False
        only_nfiles = None
        if "force" in kwargs:
            force = kwargs.pop("force")
        if "append" in kwargs:
            append = kwargs.pop("append")
        if "only_nfiles" in kwargs:
            only_nfiles = kwargs.pop("only_nfiles")
        if self.harvested:
            Logger.info("Variables have already been harvested!\
                         if you really want to reload the filelist,\
                         use 'force=True'.\
                         If you do so, all your harvested variables will be deleted!")
            if not force:
                return
            else:
                Logger.warning("..using force..")

        # FIXME: not sure if self.datasets should be
        # reinitialized with empty list here...
        datasets = {}
        if "datasets" in kwargs:
            datasets = kwargs["datasets"]
            Logger.debug("Found datasets".format(datasets))
            self.datasets = kwargs.pop("datasets")

        if datasets:
            filtered_files = []
            files = harvest_files(*args, **kwargs)
            Logger.debug("Found {} files, will start filtering...".format(len(files)))
            datasets = [self._ds_regexp(x) for x in files]
            assert len(datasets) == len(files)

            ds_files = list(zip(datasets, files))
            for k in list(self.datasets.keys()):
                filtered_files.extend([x[1] for x in ds_files if x[0] == k])
            files = filtered_files
        else:
            files = harvest_files(*args, **kwargs)

        if only_nfiles is not None:
            files = files[:only_nfiles]

        if append:
            self.files += files
        else:
            self.files = files

    def explore_files(self):
        """
        Get a sneak preview of what variables are avaukabke
        for readout

        Returns:
            list
        """
        tablenames = []
        if self.files is None:
            return

        for f in self.files:
            if tables.is_hdf5_file(f):
                with tables.open_file(f) as t:
                    for n in t.iter_nodes("/"):
                        thisname = "/"
                        if isinstance(n, tables.Table):
                            tablenames.append(thisname + n.name)
                        if isinstance(n, tables.Group):
                            tablenames.append("Group (unknown name): {}".format(n.__members__))
                    t.close()
            else:
                Logger.info("Can not peek into rootfiles at the moment")

        return tablenames

    def get(self, varkey, uncut=False):
        """
        Retrieve the data of a variable

        Args:
            varkey (str): The name of the variable

        Keyword Args:
            uncut (bool): never return cutted values
        """

        if varkey not in self.vardict:
            raise KeyError("{} not found!".format(varkey))

        # a single value for the parameters - no len available
        if self.vardict[varkey].role == variables.VariableRole.PARAMETER:
            return self.vardict[varkey].data

        if len(self.vardict[varkey].data) and len(self.cutmask) and not uncut:
            return self.vardict[varkey].data[self.cutmask]
        else:
            return self.vardict[varkey].data

    def read_variables(self, names=None, max_cpu_cores=MAX_CORES, dtype=np.float64):
        """
        Harvest the variables in self.vardict

        Keyword Args:
            names (list): havest only these variables
            max_cpu_cores (list): use a maximum of X cores of the cpu
            dtype (np.dtype) : Cast to this datatype (defalut np.float64)
        """

        assert self.files, "Need to assign some files before reading out variables"

        if names is None:
            names = list(self.vardict.keys())
        compound_variables = [] #harvest them later

        executor = fut.ProcessPoolExecutor(max_workers=max_cpu_cores)
        future_to_varname = {}

        # first read out variables,
        # then compound variables
        # so make sure they are in the 
        # right order
        simple_vars = []
        for varname in names:
            try:
                if isinstance(self.vardict[varname],variables.CompoundVariable):
                    compound_variables.append(varname)
                    continue

                elif isinstance(self.vardict[varname],variables.VariableList):
                    compound_variables.append(varname)
                    continue
                else:
                    simple_vars.append(varname)
            except KeyError:
                Logger.warning("Cannot find {} in variables!".format(varname))
                continue
        for varname in simple_vars:
            # FIXME: Make it an option to not use
            # multi cpu readout!
            #self.vardict[varname].data = variables.harvest(self.files,self.vardict[varname].definitions)
            direct_trafo = None
            if self.vardict[varname].role == variables.VariableRole.PARAMETER:
                # we need to apply  the transformation directly at readout
                direct_trafo = self.vardict[varname].transform
            future_to_varname[executor.submit(variables.harvest,\
                                              self.files,\
                                              self.vardict[varname].definitions,\
                                              nevents = self.vardict[varname].nevents,\
                                              dtype = dtype,\
                                              transformation = direct_trafo,\
                                              reduce_dimension = self.vardict[varname].reduce_dimension)] = varname
                                              #transformation = self.vardict[varname].transform)] = varname

        progbar = False
        try:
            import tqdm
            n_it = len(list(future_to_varname.keys()))
            #bar = pyprind.ProgBar(n_it,monitor=False,bar_char='#',title=self.name)
            if isnotebook():
                bar = tqdm.tqdm_notebook(total=n_it, desc=self.name, leave=False)
            else:
                bar = tqdm.tqdm(total=n_it, desc=self.name, leave=False)
            progbar = True
        except ImportError:
            pass

        exc_caught = """"""
        for future in fut.as_completed(future_to_varname):
            varname = future_to_varname[future]
            Logger.debug("Reading {} finished".format(varname))
            try:
                data = future.result()
                Logger.debug("Found {} entries for {}".format(len(data), varname))
            except Exception as exc:
                exc_caught += "Reading {} for {} generated an exception: {} - {}\n".format(varname,self.name,type(exc), exc)
                data = pd.Series([])

                # FIXME: check how different these two approaches really are
                #        the second does not work for some vector data
                #        from root files
            # also FIXME: in case of parameters, we apply the transformation directy when 
            # reading the file out, in case it is some object which does not support 
            # the numpy mechanism, e.g. root histogram (which is non-picklable) and needs 
            # to be transformed first.
            if not self.vardict[varname].role == variables.VariableRole.PARAMETER:
                if not (self.vardict[varname].transform is None):
                    data = data.map(self.vardict[varname].transform)
            #data = self.vardict[varname].transform(data)

            self.vardict[varname]._data = data
            self.vardict[varname].declare_harvested()
            if progbar: bar.update()
        for varname in compound_variables:
            #FIXME check if this causes a memory leak
            self.vardict[varname].rewire_variables(self.vardict)
            self.vardict[varname].harvest()

        if exc_caught:
            Logger.warning("During the variable readout some exceptions occured!\n" + exc_caught)
        self.declare_harvested()

        if progbar:
            bar.close()
            del bar

    @abc.abstractmethod
    def calculate_weights(self, model, model_args=None):
        return

    def get_datacube(self):
        cube = dict()
        for k in list(self.vardict.keys()):
            cube[k] = self.get(k)

        return pd.DataFrame(cube)

    @property
    def weights(self):
        if self._weights is None: # create on the fly
            return np.ones(len(self))

        if len(self.cutmask):
            return self._weights[self.cutmask]
        else:
            return self._weights

    @property
    def integrated_rate(self):
        """
        Calculate the total eventrate of this category
        (requires weights)

        Returns (tuple): rate and quadratic error
        """

        rate  = self.weights.sum()
        error = np.sqrt((self.weights**2).sum())
        return (rate,error)

    def add_livetime_weighted(self,other,self_livetime=None,other_livetime=None):
        """
        Combine two datasets livetime weighted. If it is simulated data,
        then in general it does not know about the detector livetime.
        In this case the livetimes for the two datasets can be given

        Args:
            other (pyevsel.categories.Category): Add this dataset

        Keyword Args:
            self_livetime (float): the data livetime for this dataset
            other_livetime (float): the data livetime for the other dataset

        """

        assert list(self.vardict.keys()) == list(other.vardict.keys()),"Must have the same variables to be combined"

        if isinstance(self,Data):
            self_livetime = self.livetime

        if isinstance(other,Data):
            other_livetime = other.livetime

        for k in list(other.datasets.keys()):
            self.datasets.update({k : other.datasets[k]})
        self.files.extend(other.files)
        if self.cuts or other.cuts:
            self.cuts.extend(other.cuts)
        if len(self.cutmask) or len(other.cutmask):
            self.cutmask = np.hstack((self.cutmask,other.cutmask))

        for name in self.variablenames:
            self.vardict[name].data = pd.concat([self.vardict[name].data,other.vardict[name].data])

        self_weight = (self_livetime/(self_livetime + other_livetime))
        other_weight = (other_livetime/(self_livetime + other_livetime))

        self._weights = pd.concat([self_weight*self._weights,other_weight*other._weights])
        if isinstance(self,Data):
            self.set_livetime(self.livetime + other.livetime)

    def add_plotoptions(self, options):
        """
        Add options on how to plot this category. If available,
        they will be used.

        Args:
            options (dict): For the names which are currently supported,
                            please see the example file
        """
        self.plot_options = options

    def show(self):
        """
        Print out the names of the loaded variables

        Returns:
            dict (name, len)
        """
        if not self.harvested:
            Logger.warn("No variables for {} loaded yet!".format(self.name))
            return {}

        lengths = {}
        for k in self.vardict.keys():
            lengths[k] = len(self.get(k))

        repr = ""
        for k in lengths:
            repr += "{} with definition {} : {} data points\n".\
                format(k, self.vardict[k].definitions[0][0], lengths[k])

        Logger.info(repr)
        return lengths

class Simulation(AbstractBaseCategory):
    """
    An interface to variables from simulated data
    Allows to weight the events
    """
    _mc_p_readout = False

    def __init__(self,name, weightvarname=None):
        """

        Args:
            name (str): An unique identifier for tis category

        Keyword Args:
            weightvarname (str): Use this variable for weighting
        """
        AbstractBaseCategory.__init__(self,name)
        self.weightvarname = weightvarname

    @property
    def mc_p_readout(self):
        mc_readout = []
        for k in self.vardict.keys():
            if k in [MC_P_EN, MC_P_TY, MC_P_ZE, MC_P_WE]:
                mc_readout.append(k)
        Logger.debug("Read out mc information for {}".format(mc_readout))
        return bool(len(mc_readout))

    def read_mc_primary(self,energy_var=MC_P_EN,\
                        type_var=MC_P_TY,\
                        zenith_var=MC_P_ZE,\
                        weight_var=MC_P_WE):
        """
        Trigger the readout of MC Primary information
        Rename variables to magic keywords if necessary

        Keyword Args:
            energy_var (str): simulated primary energy
            type_var (str): simulated primary type
            zenith_var (str): simulated primary zenith
            weight_var (str): a weight, e.g. interaction propability
        """

        self.read_variables([energy_var,type_var,zenith_var,weight_var])
        for varname,defaultname in [(energy_var, MC_P_EN),\
                                    (type_var, MC_P_TY),\
                                    (zenith_var, MC_P_ZE),
                                    (weight_var, MC_P_WE)]:
            if varname != defaultname:
                Logger.warning("..renaming {} to {}..".format(varname,defaultname))
                self.vardict[varname].name = defaultname

        self._mc_p_readout = True

    def calculate_weights(self, model=None, model_args=None):
        """
        Walk the variables of this category and identify the
        weighting variables and calculate them.

        Usage example: calculate_weights(model=lambda x: np.pow(x, -2.), model_args=["primary_energy"])

        Keyword Args:
            model (func)      : The target flux to weight to, if None, generated flux is used for weighting
            model_args (list) : The variables the model should be applied to from the variable dict

        Returns:
            np.ndarray
        """

        if model is None:
            Logger.info("No model given, will attempt automatic weighting")
            Logger.info("Will deduce weights from variable roles")
            fluxvarname = None
            generatorvarname = None
            for var in self.variablenames:
                if self.vardict[var].role == self.vardict[var].ROLES.FLUXWEIGHT:
                    if fluxvarname is not None:
                        raise ValueError(f"Fluxweights already found with {fluxvarname}. Definitiion must be unique. Can not set {var}")
                    fluxvarname = var
                    Logger.info(f"Found fluxweights {fluxvarname}")
                if self.vardict[var].role == self.vardict[var].ROLES.GENERATORWEIGHT:
                    if generatorvarname is not None:
                        raise ValueError(f"Fluxweights already found with {generatorvarname}. Definitiion must be unique. Can not set {var}")
                    generatorvarname = var
                    Logger.info(f"Found generator weight {var}")

            if fluxvarname is None:
                Logger.warning("Can not find fluxweigths, assuming unity")
                fluxweights = np.ones(self.raw_count)
            else:
                fluxweights = self.get(fluxvarname)

            if generatorvarname is None:
                Logger.warning("Can not find generatorweights, assuming unity")
                generatorweights = np.ones(self.raw_count)
            else:
                generatorweights = self.get(generatorvarname)

            self._weights = fluxweights/generatorweights
        else:
            Logger.warning("Model currently not supported")
            self._weights = None
        return

        #FIXME
        #if self.weightvarname is None:
        #    Logger.warn("Have to specify which variable to use for weighting! Set weightvarname first!")
        #    self._weights = None
        #    return
        #
        ##weights = [self.vardict[v] for v in self.vardict if self.vardict[v].role == v.ROLES.WEIGHT]
        ##weight_vars = [v for v in weights if v.name == self.weightvarname]
        ##if len(weight_vars) != 1:
        ##    Logger.warn("Can not calculate weights, {} weight variables found!".format(len(weight_vars)))
        ##    self._weights = None
        ##    return
        #if model is None:
        #    self._weights = self.vardict[self.weightvarname].data
        #else:
        #    model_args = [self.get(v) for v in model_args]
        #    target_flux = model(*model_args)
        #    self._weights = target_flux/self.vardict[self.weightvarname].data
        #

    # def get_weights(self, model=None, model_kwargs = None):
    #     """
    #     Calculate weights for the variables in this category
    #
    #     Args:
    #         model (callable): A model to be evaluated
    #
    #     Keyword Args:
    #         model_kwargs (dict): Will be passed to model
    #     """
    #
    #     # FIXME: clean up this mess
    #
    #     # inspect the argumentes and the weightfunction
    #     if not callable(model):
    #         self._weights = pd.Series(np.ones(self.raw_count, dtype=np.float16) / model)
    #         return
    #
    #     if not self.mc_p_readout:
    #         self.read_mc_primary()
    #
    #     if model_kwargs is None:
    #         model_kwargs = dict()
    #     func_kwargs = {MC_P_EN : self.get(MC_P_EN),\
    #                    MC_P_TY : self.get(MC_P_TY),\
    #                    MC_P_WE : self.get(MC_P_WE)}
    #
    #     for key in MC_P_ZE,MC_P_GW,MC_P_TS,DATASETS:
    #         reg = key
    #         if key == DATASETS:
    #             reg = 'mc_datasets'
    #         try:
    #             func_kwargs[reg] = self.get(key)
    #         except KeyError:
    #             Logger.warning("No MCPrimary {0} information! Trying to omit..".format(key))
    #
    #     func_kwargs.update(model_kwargs)
    #     Logger.info("Getting weights for datasets {}".format(self.datasets.__repr__()))
    #     self._weights = pd.Series(self._weightfunction(model, self.datasets,\
    #                                **func_kwargs))

    @property
    def livetime(self):
        if self.weights.sum() == 0:
            Logger.warning("Weightsum is zero!")
            return np.nan
        else:
            return self.weights.sum() / np.power(self.weights, 2).sum()

class ReweightedSimulation(Simulation):
    """
    A proxy for simulation dataset, when only the weighting differs
    """

    def __init__(self,name,mother):
        Simulation.__init__(self,name)
        self._mother = mother

    # proxies
    @property
    def mother(self):
        return self._mother

    setter = lambda self,other : None
    vardict       = property(lambda self: self.mother.vardict,\
                        setter)
    datasets      = property(lambda self: self.mother.datasets,\
                        setter)
    files         = property(lambda self: self.mother.files,\
                        setter)
    harvested = property(lambda self: self.mother.harvested, \
                             setter)
    _mc_p_readout = property(lambda self: self.mother.mc_p_readout,\
                             setter)

    @property
    def raw_count(self):
        return self.mother.raw_count

    def read_variables(self,names=None, max_cpu_cores=MAX_CORES, dtype=np.float64):
        return self.mother.read_variables(names=names, max_cpu_cores=max_cpu_cores, dtype=dtype)

    def read_mc_primary(self,energy_var=MC_P_EN,\
                       type_var=MC_P_TY,\
                       zenith_var=MC_P_ZE,\
                       weight_var=MC_P_WE):
        return self.mother.read_mc_primary(energy_var,type_var,zenith_var, weight_var)

    def add_livetime_weighted(self,other):
        raise ValueError('ReweightedSimulation datasets can not be combined! Instanciate after adding mothers instead!')

    def get(self,varname, uncut=False):
        data = self.mother.get(varname, uncut=True)

        if len(self.cutmask) and not uncut:
            return data[self.cutmask]
        else:
            return data


class Data(AbstractBaseCategory):
    """
    An interface to real time event data
    Simplified weighting only
    """

    def __init__(self,name):
        """
        Instanciate a Data dataset. Provide livetime in **kwargs.
        Special keyword "guess" for livetime allows to guess the livetime later on

        Args:
            name: a unique identifier

        Returns:

        """
        AbstractBaseCategory.__init__(self,name)
        self._runstartstop_set = False
        self._livetime = np.nan

    @staticmethod
    def _ds_regexp(filename):
        return EXP_RUN_ID(filename)

    def set_weightfunction(self,func):
        return

    def set_livetime(self,livetime):
        """
        Override the private _livetime member

        Args:
            livetime: The time needed for data-taking

        Returns:
             None

        """
        self._livetime = livetime

    # livetime is read-only
    @property
    def livetime(self):
        return self._livetime

    def set_run_start_stop(self,runstart_var=variables.Variable(None),runstop_var=variables.Variable(None)):
        """
        Let the simulation category know which 
        are the paramters describing the primary

        Keyword Args:
            runstart_var (pyevself.variables.variables.Variable/str): beginning of a run
            runstop_var (pyevself.variables.variables.Variable/str): beginning of a run

        """
        #FIXME
        for var,name in [(runstart_var,RUN_START),(runstop_var,RUN_STOP)]:
            if isinstance(var, str):
                var = self.get(var)

            if var.name is None:
                Logger.warning("No {0} available".format(name))
            elif name in self.vardict:
                Logger.info("..{0} already defined, skipping...".format(name))
                continue
            
            else:
                if var.name != name:
                    Logger.info("..renaming {0} to {1}..".format(var.name,name))
                    var.name = name
                newvar = deepcopy(var)
                self.vardict[name] = newvar

        self._runstartstop_set = True

    def estimate_livetime(self, force=False):
        """
        Calculate the livetime from run start/stop times, account for gaps
        
        Keyword Args:
            force (bool): overide existing livetime
        """
        if self.livetime and (not self.livetime == "guess"):
            Logger.warning("There is already a livetime of {:4.2f} ".format(self.livetime))
            if force:
                Logger.warning("Applying force...")
            else:
                Logger.warning("If you really want to do this, use force = True")
                return
        
        if not self._runstartstop_set:
            if (RUN_STOP in list(self.vardict.keys())) and (RUN_START in list(self.vardict.keys())):
                self._runstartstop_set = True
            else:
                Logger.warning("Need to set run start and stop times first! use object.set_run_start_stop")
                return

        Logger.warning("This is a crude estimate! Rather use a good run list or something!")
        lengths = self.get(RUN_STOP) - self.get(RUN_START)
        gaps    = self.get(RUN_START)[1:] - self.get(RUN_STOP)[:-1] #trust me!
        #h = self.nodes["header"].read()
        #h0 = h[:-1]
        #h1 = h[1:]
        ##FIXME
        #lengths = ((h["time_end_mjd_day"] - h["time_start_mjd_day"]) * 24. * 3600. +
        #           (h["time_end_mjd_sec"] - h["time_start_mjd_sec"]) +
        #           (h["time_end_mjd_ns"] - h["time_start_mjd_ns"])*1e-9 )
 
        #gaps = ((h1["time_start_mjd_day"] - h0["time_end_mjd_day"]) * 24.  * 3600. +
        #        (h1["time_start_mjd_sec"] - h0["time_end_mjd_sec"]) +
        #        (h1["time_start_mjd_ns"] - h0["time_end_mjd_ns"])*1e-9)
 

        # detector livetime is the duration of all events + the length of      all
        # gaps between events that are short enough to be not downtime. (     guess: 30s)
        est_ltime =  ( lengths.sum() + gaps[(0<gaps) & (gaps<30)].sum() )
        self.set_livetime(est_ltime)
        return 

    def calculate_weights(self, model=None, model_args=None):
        """
        Calculate weights as rate, that is number of
        events per livetime

        Keyword Args: for compatibility...
        """

        #self.set_livetime(livetime)
        if self.livetime == "guess":
            self.estimate_livetime()
        self._weights = pd.Series(np.ones(self.raw_count, dtype=np.float64)/self.livetime)


class CombinedCategory(object):
    """
    Create a combined category out of several others
    This is mainly useful for plotting
    FIXME: should this inherit from category as well?
    The difference compared to the dataset is that
    this is flat
    """

    def __init__(self, name, categories):
        self.name = name
        self.categories = categories
        self.plot = True
        self.plot_options = dict()
        self.show_in_table = True

    @property
    def weights(self):
        return pd.concat([pd.Series(cat.weights) for cat in self.categories])

    @property
    def vardict(self):
        return self.categories[0].vardict

    def get(self, varname):
        return pd.concat([cat.get(varname) for cat in self.categories])

    @property
    def integrated_rate(self):
        """
        Calculate the total eventrate of this category
        (requires weights)

        Returns (tuple): rate and quadratic error
        """

        rate = self.weights.sum()
        error = np.sqrt((self.weights ** 2).sum())
        return (rate, error)

    def add_plotoptions(self, options):
        """
        Add options on how to plot this category. If available,
        they will be used.

        Args:
            options (dict): For the names which are currently supported,
                            please see the example file
        """
        self.plot_options = options
