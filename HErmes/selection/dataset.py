"""
Provides a container for different
categories
"""
from __future__ import division

import pandas as pd
import numpy as np

from collections import OrderedDict

from ..plotting import VariableDistributionPlot
from ..utils.logger import Logger
from dashi.tinytable import TinyTable

from builtins import map
from builtins import object
from . import categories
from . import variables

from copy import deepcopy as copy


def get_label(category):
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
    of them
    """

    def __init__(self, *args, **kwargs):
        """
        Iniitalize with the categories

        Args:
            *args: pyevsel.variables.categories.Category list

        Keyword Args:
            combined_categories: 

        Returns:

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

    def add_variable(self, variable):
        """
        Add a variable to this category

        Args:
            variable (pyevsel.variables.variables.Variable): A Variable instalce
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

    #@GetTiming
    def read_variables(self, names=None):
        """
        Read out the variable for all categories

        Keyword Args:
            names (str): Readout only these variables if given
        Returns:

        """
        for cat in self.categories:
            Logger.debug("Reading variables for {}".format(cat))
            cat.read_variables(names=names)

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


    def get_weights(self, models):
        """
        Calculate the weights for all categories

        Args:
            models (dict or callable): A dictionary of categoryname -> model or a single clbl
        """
        if isinstance(models, dict):
            for catname in models:
                self.get_category(catname).get_weights(models[catname])
        if callable(models):
            for cat in self.categories:
                cat.get_weights(models)

    def add_category(self,category):
        """
        Add another category to the dataset

        Args:
            category (pyevsel.categories.Category): add this category

        """

        self.categories.append(category)

    def get_category(self,categoryname):
        """
        Get a reference to a category

        Args:
            category: A name which has to be associated to a category

        Returns (pyevsel.variables.categories.Category): Category
        """

        for cat in self.categories:
            if cat.name == categoryname:
                return cat

        raise KeyError("Can not find category {}".format(categoryname))

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
            w[cat.name] = cat.weights
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
            cut (pyevsel.variables.cut.Cut): Append this cut to the internal cutlist

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

    def get_sparsest_category(self,omit_zeros=True):
        """
        Find out which category of the dataset has the least statiscal power

        Keyword Args:
            omit_zeros (bool): if a category has no entries at all, omit
        Returns:
            str: category name
        """

        name  = self.categories[0].name
        count = self.categories[0].raw_count
        for cat in self.categories:
            if cat.raw_count < count:
                if (cat.raw_count == 0) and omit_zeros:
                    continue
                count = cat.raw_count
                name  = cat.name
        return name

    def plot_distribution(self,name,\
                          ratio=([],[]),
                          cumulative=True,
                          axes_locator=((0,"c"),(1,"r"),(2,"h")),
                          heights=(.4,.2,.2),
                          color_palette='dark',
                          normalized = False,
                          styles = dict(),
                          savepath="",savename="vdistplot"):
        """
        One shot short-cut for one of the most used
        plots in eventselections

        Args:
            name (string): The name of the variable to plot

        Keyword Args:
            path (str): The path under which the plot will be saved.
            ratio (list): A ratio plot of these categories will be crated
            color_palette (str): A predifined color palette (from seaborn or plotcolors.py
            normalized (bool): Normalize the histogram by number of events
            styles (dict): plot styling options
        Returns:
            pyevsel.variables.VariableDistributonPlot
        """
        cuts = self.categories[0].cuts
        sparsest = self.get_sparsest_category()

        bins = self.get_category(sparsest).vardict[name].calculate_fd_bins()
        plot = VariableDistributionPlot(cuts=cuts,bins=bins)
        if styles:
            plot.plot_options = styles
        else:
            plot.plot_options = self.default_plotstyles
        plotcategories = self.categories + self.combined_categories 
        for cat in [x for x in plotcategories if x.plot]:
            plot.add_variable(cat,name)
            if cumulative:
                plot.add_cumul(cat.name)
        if len(ratio[0]) and len(ratio[1]):
            tratio,tratio_err = self.calc_ratio(nominator=ratio[0],\
                                            denominator=ratio[1])

            plot.add_ratio(ratio[0],ratio[1],total_ratio=tratio,total_ratio_errors=tratio_err)
        plot.plot(axes_locator=axes_locator,\
                  heights=heights, normalized=normalized)
        #plot.add_legend()
        plot.canvas.save(savepath,savename,dpi=350)
        return plot

    @property
    def integrated_rate(self):
        """
        Integrated rate for each category

        Returns (pandas.Panel):
            rate with error
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