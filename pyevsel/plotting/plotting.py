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
from pyevsel.plotting import GetCategoryConfig,LoadConfig
import pylab as p

STD_CONF=os.path.join(os.path.split(__file__)[0],"plotsconfig.yaml")



STYLE = os.path.join(os.path.split(__file__)[0],"pyevseldefault.mplstyle")
p.style.use(STYLE)

###############################################

def CreateArrow(x_0,y_0,dx,dy,length,\
                width = .1,shape="right",\
                fc="k",ec="k",\
                alpha=1.,log=False):
    """
    Create an arrow object for plots

    Args:
        x_0 (float): x-origin
        y_0 (float): y-origin
        dx (float): x length
        dy (float): y length
        lenght (float): scale natural length 
    Keyword Args:
        width (float): thickness of arrow
        shape (str): either full, left or right
        fc (str): facecolor
        ec (str): edgecolor
        alpha (float): 0 -1 
        log (bool): If log, take care of proportions
    """
    head_starts_at_zero = False
    head_width = width*5
    head_length = length*0.1
    if log: # slightly bigger arrow
        head_width = width*6
        head_length = length*0.2

    arrow_params={'length_includes_head':False, 'shape':shape, 'head_starts_at_zero':head_starts_at_zero}
    arr = p.arrow(x_0, y_0, dx*length, dy*length, fc=fc, ec=ec, alpha=alpha, width=width, head_width=head_width,head_length=head_length, **arrow_params)
    return arr

###########################

def PlotVariableDistribution(categories,name,ratio=([],[])):
    """
    One shot short-cut for one of the most used
    plots in eventselections

    Args:
        categories (list): A list of categories which should be plotted
        name (string): The name of the variable to plot

    Keyword Args:
        ratio (list): A ratio plot of these categories will be crated
    """
    plot = VariableDistributionPlot()
    for cat in categories:
        plot.add_variable(cat,name)
        plot.add_cumul(cat.name)
    plot.add_ratio(ratio[0],ratio[1])
    plot.plot(heights=[.4,.2,.2])
    #plot.add_legend()
    plot.canvas.save("","deletemenow",dpi=350)
    return plot

###############################################

class VariableDistributionPlot(object):
    """
    A container class to hold histograms
    and ratio plots for variables
    """

    def __init__(self,cuts=[]):
        self.histograms = {}
        self.histratios = {}
        self.cumuls     = {}
        self.plotratio  = False
        self.plotcumul  = False
        self.canvas     = None
        self.label      = ''
        self.cuts       = cuts
        #self.categories = {}
        #for k in categories.keys():
        #    self.categories[k] = categories[k]

    def _add_data(self,dataname,variable_data,bins,weights=None,label=''):
        """
        Histogram the added data and store internally
        
        Args:
            dataname (string): the name of a category
            variable_data (array): the actual data
            bins (array): histogram binning
        
        Keyword Args:
            weights (array): weights for the histogram
        """
        if weights is None:
            weights = n.ones(len(variable_data))
        self.histograms[dataname] = d.factory.hist1d(variable_data,bins,weights=weights)
        self.label = label

    def indicate_cut(self,ax):
        """
        If cuts are given, indicate them by lines

        Args:
            ax (pylab.axes): axes to draw on

        """
        vmin,vmax = ax.get_ylim()
        hmin,hmax = ax.get_xlim()
        for __,operator,value in self.cuts:
            width = vmax/50.
            ax.vlines(value,ymin=vmin,ymax=vmax,linestyle=':')
            length = (hmax - hmin)*0.1
            shape = 'left'
            inversed = False
            if operator in ('>','>='):
                shape = 'right'
                inversed = True

            arr = CreateArrow(cutval,vmax*0.1, -1., 0, length, width= width,shape=shape,log=True)
            ax.add_patch(arr)
            if not inversed:
                ax.axvspan(value, hmax, facecolor=st.helpercolors["prohibited"], alpha=0.5)
            else:
                ax.axvspan(hmin, value, facecolor=st.helpercolors["prohibited"], alpha=0.5)

    def add_variable(self,category,variable_name):
        """
        Convenience interface if data is sorted in categories already

        Args:
           category (pyevsel.variables.category.Category): Get variable from this category
           variable_name (string): The name of the variable

        """
        self._add_data(category.name,category.get(variable_name),category.vardict[variable_name].bins,weights=category.weights,label=category.vardict[variable_name].label)

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
        """
        Add a cumulative distribution to the plto

        Args:
            name (str): the name of the category
        """
        self.cumuls[name] = self.histograms[name].normalized()
        

    def _draw_distribution(self,ax,name,log=True,cumulative=False,configfilename=STD_CONF,color_palette="dark"):
        """
        Paint the histograms!
        """
        color_palette = GetColorPalette(color_palette)
        cfg = GetCategoryConfig(name)
        color = cfg["dashistyle"]["color"]
        if isinstance(color,int):
            color = color_palette[color]
        cfg["dashistyle"]["color"] = color
        if cumulative:
            histograms = self.cumuls
            log = False # no log, no way!
        else:
            histograms = self.histograms

        if cfg['histscatter'] == 'scatter':
            histograms[name].scatter(log=log,cumulative=cumulative,label=cfg["label"],**cfg["dashistylescatter"])
        elif cfg['histscatter'] == "line":
            histograms[name].line(log=log,cumulative=cumulative,label=cfg["label"],**cfg["dashistyle"])
        elif cfg['histscatter'] == "overlay":
            histograms[name].line(log=log,cumulative=cumulative,label=cfg["label"],**cfg["dashistyle"])
            histograms[name].scatter(log=log,cumulative=cumulative,**cfg["dashistylescatter"])
        if cumulative:
            ax.set_ylabel('fraction')
        else:
            ax.set_ylabel('rate/bin [1/s]')

    def _draw_histratio(self,name,axes,ylim=(0.1,2.5)):
        """
        Plot on of the ratios
        """
        ratio,total_ratio,total_ratio_errors,label = self.histratios[name]
        ratio.scatter(c="k",marker="o",markersize=3)
        axes.hlines(total_ratio,axes.get_xlim()[0],axes.get_xlim()[1],linestyle="--")
        if total_ratio_errors is not None:
            axes.hlines(total_ratio + total_ratio_errors,axes.get_xlim()[0],axes.get_xlim()[1],linestyle=":")
            axes.hlines(total_ratio - total_ratio_errors,axes.get_xlim()[0],axes.get_xlim()[1],linestyle=":")
            xs = n.linspace(axes.get_xlim()[0],axes.get_xlim()[1],200)
            axes.fill_between(xs,total_ratio - total_ratio_errors, total_ratio + total_ratio_errors,facecolor="grey",alpha=0.3)
        axes.set_ylim(ylim)
        axes.set_ylabel(label)
        axes.grid(1)

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

    def plot(self,heights=(.5,.2,.2),\
             axes_locator=((0,"c"),(1,"r"),(2,"h")),\
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
                    self._draw_distribution(cur_ax,k,cumulative=True,log=log)
                break
            else:
                k = self.cumuls[self.cumuls.keys()[ax[0]]]
                self._draw_distribution(cur_ax,cumulative=True,log=log)    
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
                    print "drawing..",k
                    self._draw_distribution(cur_ax,k,log=log)
                break
            else:
                k = self.histograms[self.histograms.keys()[ax[0]]]
                self._draw_distribution(cur_ax,k,log=log)    
        lgax = self.canvas.select_axes(-1)#most upper one
        lg = lgax.legend(**LoadConfig()['legend'])
        legendwidth = LoadConfig()
        legendwidth = legendwidth['legendwidth']
        lg.get_frame().set_linewidth(legendwidth)
        # cleanup
        #self.canvas.limit_yrange()
        self.canvas.eliminate_lower_yticks()
        # set the label on the lowest axes
        self.canvas.axes[0].set_xlabel(self.label)


    def add_legend(self,**kwargs):
        """
        Add a legend to the plot
        
        Keyword Args:
             will be passed to pylab.legend
        """
        if not kwargs:
            kwargs = {"bbox_to_anchor" :(0.,1.0, 1., .102), "loc" : 3, "ncol" :3, "mode" :"expand", "borderaxespad":0., "handlelength": 2,"numpoints" :1}
        self.canvas.global_legend(**kwargs)
 


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
