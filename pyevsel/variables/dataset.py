"""
Provides a container for different
categories
"""

import pandas as pd
import numpy as np

from pyevsel.plotting.plotting import VariableDistributionPlot
from pyevsel.plotting import GetCategoryConfig
from pyevsel.utils import GetTiming

from dashi.tinytable import TinyTable

import categories

class Dataset(object):
    """
    Holds different categories, relays calls to each
    of them
    """

    def __init__(self,*args):
        """
        Iniitalize with the categories

        Args:
            *args: pyevsel.variables.categories.Category list

        Returns:

        """
        self.categories = []
        for cat in args:
            self.categories.append(cat)
            self.__dict__[cat.name] = cat

    @GetTiming
    def read_all_vars(self,variable_defs):
        """
        Read out the variable for all categories

        Args:
            variable_defs: A python module containing variable definitions

        Returns:

        """
        for cat in self.categories:
            cat.load_vardefs(variable_defs)
            cat.read_variables()

    def set_weightfunction(self,weightfunction=lambda x:x):
        """
        Defines a function which is used for weighting

        Args:
            weightfunction (func or dict): if func is provided, set this to all categories
                                           if needed, provide dict, cat.name -> func for individula setting

        Returns:
            None
        """
        if isinstance(weightfunction,dict):
            for cat in self.categories:
                cat.set_weightfunction(weightfunction[cat.name])

        else:
            for cat in self.categories:
                cat.set_weightfunction(weightfunction)

    def get_weights(self,models):
        """
        Calculate the weights for all categories

        Args:
            weightfunction (func): set func used for medel weight calculation
            models (dict): A dictionary of categoryname -> model
        """
        for catname in models:
            self.get_category(catname).get_weights(models[catname])

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

        raise KeyError("Can not find category %s" %categoryname)

    def get_variable(self,varname):
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

        df = pd.DataFrame.from_dict(var,orient="index")
        return df

    @property
    def weights(self):
        w = dict()
        for cat in self.categories:
            w[cat.name] = cat.weights
        df = pd.DataFrame(w)
        return df

    def __repr__(self):
        """
        String representation
        """

        rep = """ <Dataset: """
        for cat in self.categories:
            rep += "%s " %cat.name
        rep += ">"

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

    def plot_distribution(self,name,\
                          ratio=([],[]),
                          cumulative=True,
                          heights=(.4,.2,.2),
                          savepath="",savename="vdistplot"):
        """
        One shot short-cut for one of the most used
        plots in eventselections

        Args:
            name (string): The name of the variable to plot

        Keyword Args:
            ratio (list): A ratio plot of these categories will be crated

        Returns:
            pyevsel.variables.VariableDistributonPlot
        """
        plot = VariableDistributionPlot()
        for cat in self.categories:
            plot.add_variable(cat,name)
            if cumulative:
                plot.add_cumul(cat.name)
        if len(ratio[0]) and len(ratio[1]):
            plot.add_ratio(ratio[0],ratio[1])
        plot.plot(heights=heights)
        #plot.add_legend()
        plot.canvas.save("",savename,dpi=350)
        return plot

    @property
    def integrated_rate(self):
        """
        Integrated rate for each category

        Returns (pandas.Panel): rate with error
        """

        rdata,edata,index = [],[],[]
        for cat in self.categories:
            rate,error = cat.integrated_rate
            rdata.append(rate)
            index.append(cat.name)
            edata.append(error)

        rate = pd.Series(rdata,index)
        err  = pd.Series(edata,index)
        return (rate,err)

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
        rate,error = categories[0].integrated_rate
        error = error**2
        for cat in categories[1:]:
            tmprate,tmperror = cat.integrated_rate
            rate  += tmprate # categories should be independent
            error += tmperror**2
        return (rate,np.sqrt(error))

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
        for cat in self.categories:
            if isinstance(cat,categories.Data):
                datacats.append(cat)
        if datacats:
            simcats = [cat for cat in self.categories if cat.name not in [kitty.name for kitty in datacats]]
            simrate, simerror = self.sum_rate(simcats)

        fudges = dict()
        for cat in datacats:
            rate,error = cat.integrated_rate
            try:
                fudges[cat.name] = (rate/simrate,error/simerror)
            except ZeroDivisionError:
                fudges[cat.name] = np.NaN
        rate_dict = dict()
        all_fudge_dict = dict()
        for catname in self.categorynames:
            cfg = GetCategoryConfig(catname)
            label = cfg["label"]
            rate_dict[label] = (rates[catname],errors[catname])
            if catname in fudges:
                all_fudge_dict[label] = fudges[catname]
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
                    format="html"):
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
            return "%4.2e +- %4.2e" %(input[0],input[1])

        #FIXME: sort the table columns
        rates,fudges = self._setup_table_data(signal=signal,background=background)
        tt = TinyTable()
        tt.add("Rate (1/s)", **rates)
        tt.add("Ratio",**fudges)
        return tt.render(layout=layout,format=format,format_cell=cellformatter)

    def __len__(self):
        #FIXME: to be implemented
        return None
