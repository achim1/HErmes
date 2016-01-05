"""
Some basic plot routines for plots which occur over and 
over in eventselections
"""

import yaml
import os
import os.path

import numpy as n
import dashi as d
d.visual()

from pyevsel.plotting.plotcolors import GetColorPalette
import pyevsel.plotting.canvases as c
from pyevsel.utils.logger import Logger
reload(c)

STD_CONF=os.path.join(os.path.split(__file__)[0],"plotsconfig.yaml")

import pylab as p

style = os.path.join(os.path.split(__file__)[0],"pyevseldefault.mplstyle")
p.style.use(style)

def WritePlotsConfig(config,filename=""):
    """
    """
    raise NotImplementedError

def GetConfig(name,filename=STD_CONF):
    """
    Get a certain config from a file
    """

    configs = yaml.load(open(filename,"r"))
    for cfg in configs["categories"]:
        if cfg["name"] == name:
            return cfg
    Logger.warning("No config for %s found!" %name)
    return cfg


################################################

def PlotVariableDistribution(categories,name,ratio=([],[])):
    plot = VariableDistributionPlot()
    for cat in categories:
        plot.add_variable(cat,name)
        plot.add_cumul(cat.name)
    plot.add_ratio(ratio[0],ratio[1])
    plot.plot()
    plot.canvas.save("","deletemenow")
    return plot.canvas.show()

###############################################

class VariableDistributionPlot(object):
    """
    Plot the typical 3 fold variable
    distribution plot
    """

    def __init__(self):
        self.histograms = {}
        self.histratios = {}
        self.cumuls     = {}
        self.labels     = {}
        self.plotratio  = False
        self.plotcumul  = False
        self.canvas     = None
        #self.categories = {}
        #for k in categories.keys():
        #    self.categories[k] = categories[k]

    def _add_data(self,dataname,variable_data,bins,weights=None,label="_nolegend_"):
        if weights is None:
            weights = n.ones(len(variable_data))
        self.histograms[dataname] = d.factory.hist1d(variable_data,bins,weights=weights)
        self.labels[dataname] = label

    def add_variable(self,category,variable_name):
        """
        Convenience interface if data is sorted in categories
        allready
        """
        self._add_data(category.name,category.get(variable_name),category.vardict[variable_name].bins,weights=category.weights)

    def add_ratio(self,names_upper,names_under,total_ratio=None,total_ratio_errors=None,log=False,label="data/$\Sigma$ bg"):
        """
        Add a ratio plot to the canvas
        """
        if not isinstance(names_upper,list):
            names_upper = [names_upper]
        if not isinstance(names_under,list):
            names_under = [names_under]

        name = "".join(names_upper) + "_" + "".join(names_under)
        first_upper = names_upper.pop()
        upper_hist = self.histograms[first_upper]
        upper_ws   = self.histograms[first_upper].stats.weightsum
        for name in names_upper:
            upper_hist += self.histograms[name] 
            upper_ws   += self.histograms[name].stats.weightsum
        first_under = names_under.pop()
        under_hist = self.histograms[first_under]
        under_ws = self.histograms[first_under].stats.weightsum
        for name in names_under:
            under_hist += self.histograms[name]
            under_ws   += self.histograms[name].stats.weightsum
    
        upper_hist.normalized()
        under_hist.normalized()
        ratio = d.histfuncs.histratio(upper_hist,under_hist,\
                                      log=False,ylabel=label)
        if total_ratio is None:
            total_ratio = upper_ws/under_ws
            Logger.info("Calculated scaler ratio of %4.2f from histos" %total_ratio)

        ratio.y[ratio.y > 0] = ratio.y[ratio.y > 0] + total_ratio -1
        self.histratios[name] = (ratio,total_ratio,total_ratio_errors,label)
        return name

    def add_cumul(self,name):
        self.cumuls[name] = self.histograms[name].normalized()
        

    def _draw_distribution(self,name,log=True,cumulative=False,configfilename=STD_CONF,color_palette="dark"):
        """
        Paint the histograms!
        """
        color_palette = GetColorPalette(color_palette)
        cfg = GetConfig(name,filename=configfilename)
        print cfg
        color = cfg["dashistyle"]["color"]
        if isinstance(color,int):
            color = color_palette[color]
        cfg["dashistyle"]["color"] = color
        if cfg['histscatter'] == 'scatter':
            self.histograms[name].scatter(log=log,cumulative=cumulative,label=self.labels[name],**cfg["dashistylescatter"])
        elif cfg['histscatter'] == "line":
            self.histograms[name].line(log=log,cumulative=cumulative,label=self.labels[name],**cfg["dashistyle"])
        elif cfg['histscatter'] == "overlay":
            self.histograms[name].line(log=log,cumulative=cumulative,label=self.labels[name],**cfg["dashistyle"])
            self.histograms[name].scatter(log=log,cumulative=cumulative,label="_nolegend_",**cfg["dashistylescatter"])
        if cumulative:
            self.cumuls.pop(name)
        else:
            self.histograms.pop(name)

    def _draw_histratio(self,name,axes,ylim=(0.1,2.5)):
        """
        Plot on of the ratios
        """
        ratio,total_ratio,total_ratio_errors,label = self.histratios[name]
        ratio.scatter(c="k",marker="o",markersize=3)
        axes.hlines(total_ratio,axes.get_xlim()[0],axes.get_xlim()[1],linestyle="--")
        if total_ratio_errors is not None:
            axes.hlines(total_ratio + total_ratio_errors,theax.get_xlim()[0],theax.get_xlim()[1],linestyle=":")
            axes.hlines(total_ratio - total_ratio_errors,theax.get_xlim()[0],theax.get_xlim()[1],linestyle=":")
            xs = n.linspace(axes.get_xlim()[0],theax.get_xlim()[1],200)
            axes.fill_between(xs,total_ratio - total_ratio_errors, total_ratio + total_ratio_errors,facecolor="grey",alpha=0.3)
            axes.set_ylim(ylim)
            axes.set_ylabel(label)
            axes.grid(1)
        self.histratios.pop(name)

    def _locate_axes(self,combined_cumul,combined_ratio,combined_distro):
        axes_locator = []
        if self.cumuls:
            if combined_cumul:
                axes_locator.append((0,"c"))
            else:
                axes_locator += [(x,"c") for x in range(len(self.cumuls))]

        if self.ratios:
            if combined_ratio:
                if axes_locator:
                    axes_locator.append((axes_locator[-1] + 1,"r"))
                else:
                    axes_locator.append((0,"r"))
            else:
                if axes_locator:
                    axes_locator +=[(x+ len(axes_locator),"r") for x  in range(len(self.ratios))]

        if self.histograms:
            if combined_distro:
                if axes_locator:
                    axes_locator.append((axes_locator[-1] + 1,"h"))
                else:
                    axes_locator.append((0,"h"))
            else:
                if axes_locator:
                    axes_locator +=[(x+ len(axes_locator),"h") for x  in range(len(self.histograms))]
        return axes_locator

    def plot(self,heights=[.5,.2,.2],\
             axes_locator=[(0,"c"),(1,"r"),(2,"h")],\
             combined_distro=True,\
             combined_ratio=True,\
             combined_cumul=True,
             log=True):

        Logger.info("Found %i distributions" %len(self.histograms))
        Logger.info("Found %i ratios" %len(self.histratios))
        Logger.info("Found %i cumulative distributions" %len(self.cumuls))
        if not axes_locator:
            axes_locator = self._locate_axes(combined_cumul,combined_ratio,combined_distro)

        # calculate the amount of needed axes
        assert len(axes_locator) == len(heights), "Need to specify exactly as many heights as plots you want to have"

        self.canvas = c.YStackedCanvas(axeslayout=heights)
        
        cu_axes = filter(lambda x : x[1] == "c",axes_locator)
        h_axes = filter(lambda x : x[1] == "h",axes_locator)
        r_axes = filter(lambda x : x[1] == "r",axes_locator)
        for ax in cu_axes:
            cur_ax = self.canvas.select_axes(ax[0])
            if combined_cumul:
                for k in self.cumuls.keys():
                    self._draw_distribution(k,cumulative=True,log=log)
                break
            else:
                k = self.cumuls[self.cumuls.keys()[ax[0]]]
                self._draw_distribution(k,cumulative=True,log=log)    
        for ax in r_axes:
            cur_ax = self.canvas.select_axes(ax[0])
            if combined_ratio:
                for k in self.histratios.keys():
                    self._draw_histratio(k,cur_ax)
                break
            else:
                k = self.histratios[self.histratios.keys()[ax[0]]]
                self._draw_histratio(k,cur_ax)    

        for ax in h_axes:
            cur_ax = self.canvas.select_axes(ax[0])
            if combined_distro:
                for k in self.histograms.keys():
                    self._draw_distribution(k,log=log)
                    break
            else:
                k = self.cumuls[self.histograms.keys()[ax[0]]]
                self._draw_distribution(k,log=log)    
        


#
#class PlotConfigLoader(dict):
#    """
#    Extract a configuration
#    from a predefined plotconfig
#    file
#    """
#
#    def __init__(self,filename):
#        dict.__init__(self)
#        self.filename = filename
#        data = yaml.loads(open(filename,"w"))
#        for d in data:
#            self[d["name"]] = {}
#
#    def get_dashidict(self,)
#
