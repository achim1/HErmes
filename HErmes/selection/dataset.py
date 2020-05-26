"""
Datasets group categories together. Method calls on datasets invoke the individual methods
on the individual categories. Cuts applied to datasets will act on each individual category.

"""

import pandas as pd
import numpy as np

from collections import OrderedDict
from copy import deepcopy as copy

from ..visual import VariableDistributionPlot
from ..utils import isnotebook
from ..utils import Logger
from dashi.tinytable import TinyTable

from . import categories

def get_label(category):
    """
    Get the label for labeling plots from a datasets plot_options dictionary.

    Args:
        category (HErmes.selection.categories.category): Query the category's plot_options dict, if not fall back to category.name

    Returns:
        string
    """

    if category.plot_options:
        if "label" in category.plot_options:
            return category.plot_options["label"]
        else:
            return category.name
    else:
        return category.name


class Dataset(object):
    """
    Holds different categories, relays calls to each
    of them.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            *args: HErmes.selection.variables.categories.Category list

        Keyword Args:
            combined_categories:
        """
        self.categories = []
        self.combined_categories = []
        # sort categories, do reweighted simulation last
        # FIXME: if not, there will be problems
        # FIXME: investigate!
        reweighted_categories = []

        self.default_plotstyles = {}
        for cat in args:
            self.__dict__[cat.name] = cat
            if isinstance(cat,categories.ReweightedSimulation):
                reweighted_categories.append(cat)
                continue 
            self.categories.append(cat)
        self.categories = self.categories + reweighted_categories
        if 'combined_categories' in kwargs:
            for name in list(kwargs['combined_categories'].keys()):
                self.combined_categories.append(categories.CombinedCategory(name,kwargs['combined_categories'][name]))

    def set_default_plotstyles(self, styledict):
        """
        Define a standard for each category how 
        it should appear in plots

        Args:
            styledict (dict)
        """
        self.default_plotstyles = styledict
        for cat in self.categorynames:
            self[cat].add_plotoptions(styledict[cat])

    def add_variable(self, variable):
        """
        Add a variable to this category

        Args:
            variable (HErmes.selection.variables.variables.Variable): A Variable instalce
        """

        for cat in self.categories:
            cat.add_variable(variable)

    def delete_variable(self, varname):
        """
        Delete a variable entirely from the dataset

        Args:
            varname (str): the name of the variable

        Returns:
            None
        """
        for cat in self.categories:
            cat.delete_variable(varname)

    def load_vardefs(self, vardefs):
        """
        Load the variable definitions from a module

        Args:
            vardefs (python module/dict): A module needs to contain variable definitions.
                                         It can also be a dictionary of categoryname->module

        """
        if isinstance(vardefs, dict):
            for k in vardefs:
                # FIXME: the way over self.__dict__ does not work
                # maybe there is something more fishy...
                if str(k) == "all":
                    for cat in self.categories:
                        cat.load_vardefs(vardefs[k])

                for cat in self.categories:
                    if cat.name == k:
                        cat.load_vardefs(vardefs[k])
                #self.__dict__[k].load_vardefs(vardefs)

        else:
            for cat in self.categories:
                cat.load_vardefs(vardefs)

    @property
    def variablenames(self):
        return {cat.name : cat.variablenames for cat in self.categories}

    @property
    def files(self):
        return {cat.name : cat.files for cat in self.categories}

    #@GetTiming
    def read_variables(self, names=None,
                             max_cpu_cores=categories.MAX_CORES,
                             dtype=np.float64):
        """
        Read out the variable for all categories

        Keyword Args:
            names (str): Readout only these variables if given
            max_cpu_cores (int): Maximum number of cpu cores which will be used
            dtype (np.dtype) : Cast to the given datatype (default is np.flaot64)
        Returns:
            None
        """
        progbar = False
        try:
            import tqdm
            n_it = len(self.categories)
            loader_string = "Loading dataset"
            if isnotebook():
                bar = tqdm.tqdm_notebook(total=n_it, desc=loader_string, leave=True)
            else:
                bar = tqdm.tqdm(total=n_it, desc=loader_string, leave=True)
            progbar = True
        except ImportError:
            pass
        for cat in self.categories:
            Logger.debug("Reading variables for {}".format(cat))
            cat.read_variables(names=names, max_cpu_cores=max_cpu_cores, dtype=dtype)
            if progbar: bar.update()

    def drop_empty_variables(self):
        """
        Delete variables which have no len

        Returns:
            None
        """

        for cat in self.categories:
            cat.drop_empty_variables()

    def set_weightfunction(self, weightfunction=lambda x:x):
        """
        Defines a function which is used for weighting

        Args:
            weightfunction (func or dict): if func is provided, set this to all categories
                                           if needed, provide dict, cat.name -> func for individula setting

        Returns:
            None
        """
        if isinstance(weightfunction, dict):
            for cat in self.categories:
                cat.set_weightfunction(weightfunction[cat.name])

        else:
            for cat in self.categories:
                cat.set_weightfunction(weightfunction)

    def calculate_weights(self, model=None, model_args=None):
        """
        Calculate the weights for all categories

        Keyword Args:
            model (dict/func) : Either a dict catname -> func or a single func
                                 If it is a single funct it will be applied to all categories
            model_args (dict/list): variable names as arguments for the function
        """
        if isinstance(model, dict):
            if not isinstance(model_args, dict):
                raise ValueError("if model is a dict, model_args has to be a dict too!")
            for catname in model:
                self.get_category(catname).calculate_weights(model=model[catname], model_args=model_args[catname])

        else:
            for cat in self.categories:
                cat.calculate_weights(model=model, model_args=model_args)

    # def get_weights(self, models):
    #     """
    #     Calculate the weights for all categories
    #
    #     Args:
    #         models (dict or callable): A dictionary of categoryname -> model or a single clbl
    #     """
    #     if isinstance(models, dict):
    #         for catname in models:
    #             self.get_category(catname).get_weights(models[catname])
    #     if callable(models):
    #         for cat in self.categories:
    #             cat.get_weights(models)

    def add_category(self,category):
        """
        Add another category to the dataset

        Args:
            category (HErmes.selection.categories.Category): add this category

        """

        self.categories.append(category)

    def __getitem__(self, item):
        """
        Shortcut for self.get_category/get_variable

        Args:
            item:

        Returns:
            HErmes.selection.variables.Variable/HErmes.selection.categories.Category
        """
        try:
            return self.get_category(item)
        except KeyError:
            pass
        try:
            return self.get_variable(item)
        except  KeyError:
            pass
        if ":" in item:
            cat, var = item.split(":")
            return self.get_category(cat).get(var)

        else:
            raise KeyError("{} can not be found".format(item))

    def get_category(self, categoryname):
        """
        Get a reference to a category.

        Args:
            category: A name which has to be associated to a category

        Returns:
             HErmes.selection.categories.Category
        """

        for cat in self.categories:
            if cat.name == categoryname:
                return cat

        raise KeyError("Can not find category {}.".format(categoryname))

    def get_variable(self, varname):
        """
        Get a pandas dataframe for all categories

        Args:
            varname (str): A name of a variable

        Returns:
            pandas.DataFrame: A 2d dataframe category -> variable
        """

        var = dict()
        for cat in self.categories:
            var[cat.name] = cat.get(varname)

        return pd.DataFrame.from_dict(var, orient="index")

    def set_livetime(self, livetime):
        """
        Define a livetime for this dataset.

        Args:
            livetime (float): Time interval the data was taken in. (Used for rate calculation)

        Returns:
            None
        """
        for cat in self.categories:
            if hasattr(cat, "set_livetime"):
                cat.set_livetime(livetime)

    @property
    def weights(self):
        """
        Get the weights for all categories in this dataset
        """
        w = dict()
        for cat in self.categories:
            w[cat.name] = pd.Series(cat.weights, dtype=np.float64)
        return pd.DataFrame.from_dict(w,orient='index')

    def __repr__(self):
        """
        String representation
        """

        rep = """ <Dataset: """
        for cat in self.categories:
            rep += "{} ".format(cat.name)
        rep += ">"
        return rep

    def add_cut(self,cut):
        """
        Add a cut without applying it yet

        Args:
            cut (HErmes.selection.variables.cut.Cut): Append this cut to the internal cutlist
        """
        for cat in self.categories:
            cat.add_cut(cut)

    def apply_cuts(self,inplace=False):
        """
        Apply them all!
        """
        for cat in self.categories:
            cat.apply_cuts(inplace=inplace)

    def undo_cuts(self):
        """
        Undo previously done cuts, but keep them so that
        they can be re-applied
        """
        for cat in self.categories:
            cat.undo_cuts()

    def delete_cuts(self):
        """
        Completely purge all cuts from this
        dataset
        """
        for cat in self.categories:
            cat.delete_cuts()

    @property
    def categorynames(self):
        return [cat.name for cat in self.categories]

    @property
    def combined_categorynames(self):
        return [cat.name for cat in self.combined_categories]

    def get_sparsest_category(self, omit_empty_cat=True):
        """
        Find out which category of the dataset has the least statistical power

        Keyword Args:
            omit_empty_cat (bool): if a category has no entries at all, omit
        Returns:
            str: category name
        """

        name  = self.categories[0].name
        count = self.categories[0].raw_count
        for cat in self.categories:
            if cat.raw_count < count:
                if (cat.raw_count == 0) and omit_empty_cat:
                    continue
                count = cat.raw_count
                name  = cat.name
        return name


    def distribution(self,name,\
                     ratio=([],[]),
                     cumulative=True,
                     log=False,
                     transform=None,
                     disable_weights=False,
                     color_palette='dark',
                     normalized = False,
                     styles = dict(),
                     style="classic",
                     ylabel="rate/bin [1/s]",
                     axis_properties=None,
                     ratiolabel="data/$\Sigma$ bg",
                     bins=None,
                     external_weights=None,
                     savepath=None,
                     figure_factory=None,
                     zoomin=False,
                     adjust_ticks = lambda x  : x):
        """
        One shot short-cut for one of the most used
        plots in eventselections.

        Args:
            name                 (string) : The name of the variable to plot

        Keyword Args:
            path                    (str) : The path under which the plot will be saved.
            ratio                  (list) : A ratio plot of these categories will be crated
            color_palette           (str) : A predifined color palette (from seaborn or HErmes.plotting.colors) 
            normalized             (bool) : Normalize the histogram by number of events
            transform          (callable) : Apply this transformation before plotting
            disable_weights        (bool) : Disable all weighting to avoid problems with uneven sized arrays
            styles                 (dict) : plot styling options
            ylabel                  (str) : general label for y-axis
            ratiolabel              (str) : different label for the ratio part of the plot
            bins             (np.ndarray) : binning, if None binning will be deduced from the variable definition
            figure_factory         (func) : factory function which return a matplotlib.Figure
            style                (string) : TODO "modern" || "classic" || "modern-cumul" || "classic-cumul"
            savepath             (string) : Save the canvas at given path. None means it will not be saved.
            external_weights       (dict) : supply external weights - this will OVERIDE ANY INTERNALLY CALCULATED WEIGHTS
                                            and use the supplied weights instead.
                                            Must be in the form { "categoryname" : weights}
            axis_properties        (dict) : Manually define a plot layout with up to three axes.
                                            For example, it can look like this:
                                            {
                                                "top": {"type": "h", # histogram
                                                        "height": 0.4, # height in percent
                                                        "index": 2}, # used internally
                                                "center": {"type": "r", # ratio plot
                                                            "height": 0.2,
                                                            "index": 1},
                                                "bottom": { "type": "c", # cumulative histogram
                                                            "height": 0.2,
                                                            "index": 0}
                                            }

            zoomin                 (bool) : If True, select the yrange in a way that the interesting part of the 
                                            histogram is shown. Caution is needed, since this might lead to an
                                            overinterpretation of fluctuations.
            adjust_ticks            (fcn) : A function, applied on a matplotlib axes
                                           which will set the proper axis ticks
        Returns:
            HErmes.selection.variables.VariableDistributionPlot
        """
        
        
        # if (not cumulative) or ratio  == ([],[]):
        #
        #     # assuming a single cumulative axis
        #     tmp_axis_properties = dict()
        #     unassigned_height = 0
        #
        #     for key in axis_properties:
        #         if ("c" == axis_properties[key]["type"]) and (not cumulative):
        #             unassigned_height += axis_properties[key]["height"]
        #             continue
        #         if ("r" == axis_properties[key]["type"]) and (ratio == ([],[])):
        #             unassigned_height += axis_properties[key]["height"]
        #             continue
        #
        #         tmpdict = copy(axis_properties[key])
        #         tmpdict["index"] = tmpdict["index"] -1 - bool(ratio == ([],[]))
        #         tmp_axis_properties.update({key : tmpdict})
        #
        #     n_plots = len(tmp_axis_properties.keys())
        #     extra_height = unassigned_height/float(n_plots)
        #     for key in tmp_axis_properties:
        #         tmp_axis_properties[key]["height"] += extra_height
        #
        # else:
        #     tmp_axis_properties = copy(axis_properties)

        if axis_properties is not None:
            tmp_axis_properties = copy(axis_properties)

        else:
            # always have the histogram, but add
            # cumulative or ratio plot
            if cumulative and ratio != ([],[]):
                tmp_axis_properties = {\
                    "top": {"type": "h", \
                            "height": 0.4, \
                            "index": 2},\
                    "center": {"type": "r",\
                               "height": 0.2,\
                               "index": 1},\
                    "bottom": {"type": "c", \
                               "height": 0.2,\
                               "index": 0}\
                    }
            elif cumulative:
                tmp_axis_properties = { \
                    "top": {"type": "h", \
                            "height": 0.6, \
                            "index": 1}, \
                    "bottom": {"type": "c", \
                               "height": 0.4, \
                               "index": 0} \
                    }
            elif ratio != ([],[]):
                tmp_axis_properties = { \
                    "top": {"type": "h", \
                            "height": 0.6, \
                            "index": 1}, \
                    "bottom": {"type": "r", \
                               "height": 0.4, \
                               "index": 0} \
                    }
            else:
                tmp_axis_properties = { \
                    "top": {"type": "h", \
                            "height": 0.95, \
                            "index": 0}, \
                    }
        axes_locator = [(tmp_axis_properties[k]["index"], tmp_axis_properties[k]["type"], tmp_axis_properties[k]["height"])\
                        for k in tmp_axis_properties]
        #print (axes_locator)
        #heights = [axis_properties[k]["height"] for k in axis_properties]
        cuts = self.categories[0].cuts
        sparsest = self.get_sparsest_category()

        # check if there are user-defined bins for that variable
        if bins is None:
            bins = self.get_category(sparsest).vardict[name].bins
        # calculate the best possible binning
        if bins is None:
            bins = self.get_category(sparsest).vardict[name].calculate_fd_bins()
        label = self.get_category(sparsest).vardict[name].label
        plot = VariableDistributionPlot(cuts=cuts, bins=bins,\
                                        xlabel=label,\
                                        color_palette=color_palette)
        if styles:
            plot.plot_options = styles
        else:
            plot.plot_options = self.default_plotstyles
        plotcategories = self.categories + self.combined_categories 

        Logger.warn("For variables with different lengths the weighting is broken. If weights, it will fail")
        for cat in [x for x in plotcategories if x.plot]:
            if external_weights is not None:
                weights = external_weights[cat.name]
            elif ((cat.weights is not None) and (not disable_weights)):
                weights = cat.weights
                Logger.debug(f"Found {len(weights)} weights")
                if not len(weights):
                    weights = None
            else:
                weights = None
            Logger.debug(f"Adding variable data {name}")
            plot.add_variable(cat, name, transform=transform, external_weights=weights)
            if cumulative:
                Logger.debug("Adding variable data {} for cumulative plot".format(name))
                plot.add_cumul(cat.name)

        if len(ratio[0]) and len(ratio[1]):
            Logger.debug("Requested to plot ratio {} {}".format(ratio[0], ratio[1]))
            tratio,tratio_err = self.calc_ratio(nominator=ratio[0],\
                                                denominator=ratio[1])

            plot.add_ratio(ratio[0],ratio[1],\
                           total_ratio=tratio,\
                           label=ratiolabel,
                           total_ratio_errors=tratio_err)

        plot.plot(axes_locator=axes_locator,\
                  normalized=normalized,\
                  figure_factory=figure_factory,\
                  log=log,\
                  style=style,\
                  ylabel=ylabel,\
                  zoomin=zoomin,\
                  adjust_ticks=adjust_ticks)
        #plot.add_legend()
        #plot.canvas.save(savepath,savename,dpi=350)
        if savepath is not None:
            plot.canvas.save(savepath, name)
        return plot

    @property
    def integrated_rate(self):
        """
        Integrated rate for each category

        Returns:
            pandas.Panel: rate with error
        """

        rdata,edata,index = [],[],[]
        for cat in self.categories + self.combined_categories:
            rate,error = cat.integrated_rate
            rdata.append(rate)
            index.append(cat.name)
            edata.append(error)

        rate = pd.Series(rdata,index)
        err  = pd.Series(edata,index)
        return (rate,err)

    #FIXME static method!
    def sum_rate(self,categories=None):
        """
        Sum up the integrated rates for categories

        Args:
            categories: categories considerred background

        Returns:
             tuple: rate with error

        """
        if categories is None:
            return 0,0

        categories = [self.get_category(i) if isinstance(i, str) else i for i in categories]
        rate,error = categories[0].integrated_rate
        error = error**2
        for cat in categories[1:]:
            tmprate,tmperror = cat.integrated_rate
            rate  += tmprate # categories should be independent
            error += tmperror**2
        return (rate,np.sqrt(error))

    def calc_ratio(self,nominator=None,denominator=None):
        """
        Calculate a ratio of the given categories

        Args:
            nominator (list):
            denominator (list):

        Returns:
            tuple
        """
        nominator = [self.get_category(i) if isinstance(i, str) else i for i in nominator]
        denominator = [self.get_category(i) if isinstance(i, str) else i for i in denominator]

        a,a_err = self.sum_rate(categories=nominator)
        b,b_err = self.sum_rate(categories=denominator)
        if b == 0:
            return np.nan, np.nan
        sum_err = np.sqrt((a_err/ b) ** 2 + ((-a * b_err)/ (b ** 2)) ** 2)
        return a/b, sum_err

    def _setup_table_data(self,signal=None,background=None):
        """
        Setup data for a table
        If signal and background are given, also summed values
        will be in the list

        Keyword Args:
            signal (list): category names which are considered signal
            background (list): category names which are considered background

        Returns
            dict: table dictionary
        """

        rates, errors = self.integrated_rate
        sgrate, sgerrors = self.sum_rate(signal)
        bgrate, bgerrors = self.sum_rate(background)
        allrate, allerrors = self.sum_rate(self.categories)
        tmprates  = pd.Series([sgrate,bgrate,allrate],index=["signal","background","all"])
        tmperrors = pd.Series([sgerrors,bgerrors,allerrors],index=["signal","background","all"])
        rates = rates.append(tmprates)
        errors = errors.append(tmperrors)

        datacats = []
        for cat in self.categories + self.combined_categories:
            if isinstance(cat,categories.Data):
                datacats.append(cat)
        if datacats:
            simcats = [cat for cat in self.categories if cat.name not in [kitty.name for kitty in datacats]]
            simrate, simerror = self.sum_rate(simcats)

        fudges = dict()
        for cat in datacats:
            rate,error = cat.integrated_rate
            try:
                fudges[cat.name] = (rate/simrate),(error/simerror)
            except ZeroDivisionError:
                fudges[cat.name] = np.NaN
        rate_dict = OrderedDict()
        all_fudge_dict = OrderedDict()
        #for catname in sorted(self.categorynames) + sorted(self.combined_categorynames):
        for cat in datacats:
            label = get_label(cat)
            #cfg = GetCategoryConfig(cat.name)
            #label = cfg["label"]
            rate_dict[label] = (rates[cat.name], errors[cat.name])
            if cat.name in fudges:
                all_fudge_dict[label] = fudges[cat.name]
            else:
                all_fudge_dict[label] = None

        rate_dict["Sig."] =  (rates["signal"],errors["signal"] )
        rate_dict["Bg."] = (rates["background"],errors["background"])
        rate_dict["Gr. Tot."] = (rates["all"],errors["all"])
        all_fudge_dict["Sig."]     = None
        all_fudge_dict["Bg."]      = None
        all_fudge_dict["Gr. Tot."] = None
        return rate_dict,all_fudge_dict

    def tinytable(self,signal=None,\
                    background=None,\
                    layout="v",\
                    format="html",\
                    order_by=lambda x:x,
                    livetime=1.):
        """
        Use dashi.tinytable.TinyTable to render a nice
        html representation of a rate table

        Args:
            signal (list) : summing up signal categories to calculate total signal rate
            background (list): summing up background categories to calculate total background rate
            layout (str) : "v" for vertical, "h" for horizontal
            format (str) : "html","latex","wiki"

        Returns:
            str: formatted table in desired markup
        """
        def cellformatter(input):
            #print input
            if input is None:
                return "-"
            if isinstance(input[0],pd.Series):
                input = (input[1][0],input[1][0])
            return "{:4.2e} +- {:4.2e}".format(input[0],input[1])

        #FIXME: sort the table columns
        rates,fudges = self._setup_table_data(signal=signal,background=background)
        events = dict()
        for k in rates:
            events[k] = rates[k][0] * livetime, rates[k][1] * livetime

        showcats =  [get_label(cat) for cat in self.categories if cat.show_in_table]
        showcats += [get_label(cat) for cat in self.combined_categories if cat.show_in_table]
        showcats.extend(['Sig.',"Bg.","Gr. Tot."])
        orates = OrderedDict()
        ofudges = OrderedDict()
        oevents = OrderedDict()
        for k in list(rates.keys()):
            if k in showcats:
                orates[k] = rates[k]
                ofudges[k] = fudges[k]
                oevents[k] = events[k]
        #rates  = {k : rates[k] for k in rates if k in showcats}
        #fudges = {k : fudges[k] for k in fudges if k in showcats}
        #events = {k : events[k] for k in events if k in showcats}
        tt = TinyTable()

        #bypass the add function ot add an ordered dict
        for label,data in [('Rate (1/s)', orates),("Ratio", ofudges),("Events",oevents)]:
            tt.x_labels.append(label)
            tt.label_data[label] = data
        #tt.add("Rate (1/s)", **rates)
        #tt.add("Ratio",**fudges)
        #tt.add("Events",**events)
        return tt.render(layout=layout,format=format,\
                         format_cell=cellformatter,\
                         order_by=order_by)


    #def cut_progression_table(self,cuts,\
    #                signal=None,\
    #                background=None,\
    #                layout="v",\
    #                format="html",\
    #                order_by=lambda x:x,
    #                livetime=1.):

    #    self.delete_cuts()
    #    self.undo_cuts()
    #    for cut in cuts:
    #        self.add_cut(cut)
    #        self.apply_cuts()

    def __len__(self):
        #FIXME: to be implemented
        raise NotImplementedError
