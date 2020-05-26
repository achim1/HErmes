"""
Define some
"""
from __future__ import print_function

from builtins import range
from builtins import object

from copy import deepcopy as copy

import numpy as n
import numpy as np
import dashi as d
import pylab as p

import matplotlib.ticker

from hepbasestack.colors import get_color_palette
#from .colors import get_color_palette
from .canvases import YStackedCanvas
from ..utils import Logger
from ..utils import flatten
from .. import fitting as fit

d.visual()

###############################################

def create_arrow(ax, x_0, y_0, dx, dy, length,\
                 width = .1, shape="right",\
                 fc="k", ec="k",\
                 alpha=1., log=False):
    """
    Create an arrow object for plots. This is typically a large
    arrow, which can used to indicate a region in the plot which
    is excluded by a cut.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): The axes where the arrow
                                                    will be attached to
        x_0 (float): x-origin of the arrow
        y_0 (float): y-origin of the arrow
        dx (float): x length of the arrow
        dy (float): y length of the arrow
        length (float): additional scaling parameter to scale
                        the length of the arrow

    Keyword Args:
        width (float): thickness of arrow
        shape (str): either "full", "left" or "right"
        fc (str): facecolor
        ec (str): edgecolor
        alpha (float): 0 -1 alpha value of the arrow
        log (bool): I for logscale, the proportions of the arrow will be adjusted accorginly.

    Returns:
        matplotlib.axes._subplots.AxesSubplot
    """
    head_starts_at_zero = False
    head_width = width*5
    head_length = length*0.1
    if log: # slightly bigger arrow
        head_width = width*6
        head_length = length*0.2

    arrow_params={'length_includes_head':False,\
                  'shape':shape,\
                  'head_starts_at_zero':head_starts_at_zero}
    arr = ax.arrow(x_0, y_0, dx*length,\
                   dy*length, fc=fc, ec=ec,\
                   alpha=alpha, width=width,\
                   head_width=head_width,\
                   head_length=head_length,\
                   **arrow_params)
    return ax

###############################################

def meshgrid(xs, ys):
    """
    Create x and y data for matplotlib pcolormesh and
    similar plotting functions.

    Args:
        xs (np.ndarray): 1d x bins
        ys (np.ndarray): 2d y bins

    Returns:
        tuple (np.ndarray, np.ndarray, np.ndarray): 2d X and 2d Y matrices as well as a placeholder for the Z array

    """
    xlen = len(xs)
    ylen = len(ys)
    allx, ally = [], []
    
    # prepare xs
    for __ in range(ylen):
        allx.append(xs)

    allx = np.array(allx)
    allx = allx.T

    # prepare ys
    for __ in range(xlen):
        ally.append(ys)

    ally = np.array(ally)

    zs = np.zeros([xlen, ylen])
    return allx, ally, zs

###############################################

def line_plot(quantities,
              bins=None,
              xlabel='',
              add_ratio=None,
              ratiolabel='',
              colors=None,
              figure_factory=None):
    """
    Args:
        quantities:

    Keyword Args:
        bins:
        xlabel:
        add_ratio (tuple): (["data1"],["data2"])
        ratiolabel (str):
        colors:
        figure_factory (callable): Factory function returning matplotolib.Figure

    Returns:

    """
    # FIXME XXX under development
    raise NotImplementedError

    if add_ratio is not None:
        canvas = YStackedCanvas(subplot_yheights=(0.2, 0.7),
                                space_between_plots=0.0)
        ax0 = canvas.select_axes(-1)
        data = np.array(quantities[add_ratio[0]]) / np.array(quantities[add_ratio[1]])
        data = data * 100
        thebins = np.array(bins[add_ratio[0]])
        bin_size = abs(thebins[1] - thebins[0]) / 2 * np.ones(len(thebins))
        thebins = thebins + bin_size
        ax0.plot(thebins, data, color='gray')
        ax0.scatter(thebins, data, marker='o', s=50, color='gray')
        ax0.grid(1)
        ax0.set_ylabel(ratiolabel)
        ax0.spines['top'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax = canvas.select_axes(0)
        lgax = ax0
    else:
        if figure_factory is None:
            fig = p.figure()
        else:
            fig = figure_factory()
        ax = fig.gca()
        lgax = ax

    for reconame in quantities:
        thebins = np.array(bins[reconame])
        bin_size = abs(thebins[1] - thebins[0]) / 2 * np.ones(len(thebins))
        thebins = thebins + bin_size
        label = reconame.replace('-', '')
        if colors is not None:
            ax.plot(thebins, quantities[reconame], label=label, color=colors[reconame])
            ax.scatter(thebins, quantities[reconame], marker='o', s=50, c=colors[reconame])
        else:
            ax.plot(thebins, quantities[reconame], label=label)
            ax.scatter(thebins, quantities[reconame], marker='o', s=50)

    legend_kwargs = {'bbox_to_anchor': [0.0, 1.0, 1.0, 0.102],'loc': 3,
       'frameon': False,
       'ncol': 2,
       'borderaxespad': 0,
       'mode': 'expand',
       'handlelength': 2,
       'numpoints': 1
       }
    if len(quantities.keys()) == 3:
        legend_kwargs['ncol'] = 3
    ax.grid(1)
    ax.set_ylabel('$\\cos(\\Psi)$')
    ax.set_xlabel(xlabel)
    sb.despine()
    if add_ratio:
        ax.spines['top'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.spines['left'].set_visible(False)
        legend_kwargs['bbox_to_anchor'] = [0, 1.3, 1.0, 0.102]
        ax.legend(**legend_kwargs)
        p.subplots_adjust(hspace=0.2)
        return canvas.figure
    ax.legend(**legend_kwargs)
    return fig

###############################################

def gaussian_model_fit(data,
                       startparams=(0,0.2),
                       fitrange=((None,None), (None,None)),
                       fig=None,
                       norm=True,
                       bins=80,
                       xlabel='$\\theta_{{rec}} - \\theta_{{true}}$'):
    """
    A plot with a gaussian fitted to data. A histogram of the data will be created and a gaussian
    will be fitted, with 68 and 95 percentiles indicated in the plot.

    Args:
        data (array-like)        : input data with a (preferably) gaussian distribution

    Keyword Args:
        startparams (tuple)      : a set of startparams of the gaussian fit. If only
                                   mu/sigma are given, then the plot will be normalized
        fig (matplotlib.Figure)  : pre-created figure to draw the plot in 
        bins (array-like or int) : bins for the underliying histogram
        fitrange (tuple(min, max): min-max range for the gaussian fit
        xlabel (str)             : label for the x-axes
    """
    if len(startparams) == 3:
        tofit = lambda x,mean,sigma,amp : amp*fit.gauss(x,mean,sigma)
    else:
        tofit = fit.gauss
    mod = fit.Model(tofit, startparams=startparams)
    mod.add_data(data, create_distribution=True, bins=bins, normalize=norm)
    limits = []
    notempty = False
    for k in fitrange:
        thislimit = []
        for j in k:
            if j is None:
                continue
            else:
                notempty = True
                thislimit.append(j)
        limits.append(tuple(thislimit))
    limits = tuple(limits)
    print (limits)
    if notempty:
        mod.fit_to_data(limits=limits)

    else:
        mod.fit_to_data()


    thecolors = get_color_palette()
    fig = mod.plot_result(log=False, xlabel=xlabel, add_parameter_text=(
     ('$\\mu$& {:4.2e}\\\\', 0), ('$\\sigma$& {:4.2e}\\\\', 1)), datacolor=thecolors[3], modelcolor=thecolors[3], histostyle='line', model_alpha=0.7, fig=fig)
    ax = fig.gca()
    ax.grid(1)
    ax.set_ylim(ymax=1.1 * max(mod.data))
    upper68 = mod.distribution.stats.mean + mod.distribution.stats.std
    lower68 = mod.distribution.stats.mean - mod.distribution.stats.std
    lower95 = mod.distribution.stats.mean - 2 * mod.distribution.stats.std
    upper95 = mod.distribution.stats.mean + 2 * mod.distribution.stats.std
    ax.axvspan(lower68, upper68, facecolor=thecolors[8], alpha=0.7, ec='none')
    ax.axvspan(lower95, upper95, facecolor=thecolors[8], alpha=0.3, ec='none')
    ax.text(lower68 * 0.9, max(mod.data) * 0.98, '68\\%', color=thecolors[3], fontsize=20)
    ax.text(lower95 * 0.9, max(mod.data) * 0.85, '95\\%', color=thecolors[3], fontsize=20)
    return (mod, fig)


###############################################

def gaussian_fwhm_fit(data,
                      startparams=(0,0.2,1),\
                      fitrange=((None,None), (None,None), (None, None)),\
                      fig=None,\
                      bins=80,\
                      xlabel='$\\theta_{{rec}} - \\theta_{{true}}$'):
    """
    A plot with a gaussian fitted to data. A histogram of the data will be created and a gaussian
    will be fitted, with 68 and 95 percentiles indicated in the plot. The gaussian will be in a form
    so that the fwhm can be read directly from it. The "width" parameter of the gaussian is NOT the
    standard deviation, but FWHM!

    Args:
        data (array-like)        : input data with a (preferably) gaussian distribution

    Keyword Args:
        startparams (tuple)      : a set of startparams of the gaussian fit. It is a 3
                                   parameter fit with mu, fwhm and amplitude
        fitrange (tuple)         : if desired, the fit can be restrained. One tuple of (min, max) per
                                   parameter
        fig (matplotlib.Figure)  : pre-created figure to draw the plot in
        bins (array-like or int) : bins for the underliying histogram
        xlabel (str)             : label for the x-axes
    """
    mod = fit.Model(fit.fwhm_gauss, startparams=startparams)
    mod.add_data(data, create_distribution=True, bins=bins, normalize=False)
    limits = []
    notempty = False
    for k in fitrange:
        thislimit = []
        for j in k:
            if j is None:
                continue
            else:
                notempty = True
                thislimit.append(j)
        limits.append(tuple(thislimit))
    limits = tuple(limits)
    print (limits)
    if notempty:
        mod.fit_to_data(limits=limits)

    else:
        mod.fit_to_data()

    thecolors = get_color_palette()
    fig = mod.plot_result(log=False, xlabel=xlabel, add_parameter_text=(
        ('$\\mu$& {:4.2e}\\\\', 0), ('FWHM& {:4.2e}\\\\', 1), ('AMP& {:4.2e}\\\\',2)), datacolor=thecolors[3], modelcolor=thecolors[3], histostyle='line', model_alpha=0.7, fig=fig)
    ax = fig.gca()
    ax.grid(1)
    ax.set_ylim(ymax=1.1 * max(mod.data))
    upper68 = mod.distribution.stats.mean + mod.distribution.stats.std
    lower68 = mod.distribution.stats.mean - mod.distribution.stats.std
    lower95 = mod.distribution.stats.mean - 2 * mod.distribution.stats.std
    upper95 = mod.distribution.stats.mean + 2 * mod.distribution.stats.std
    ax.axvspan(lower68, upper68, facecolor=thecolors[8], alpha=0.7, ec='none')
    ax.axvspan(lower95, upper95, facecolor=thecolors[8], alpha=0.3, ec='none')
    ax.text(lower68 * 0.9, max(mod.data) * 0.98, '68\\%', color=thecolors[3], fontsize=20)
    ax.text(lower95 * 0.9, max(mod.data) * 0.85, '95\\%', color=thecolors[3], fontsize=20)
    return (mod, fig)


###############################################

class VariableDistributionPlot(object):
    """
    A plot which shows the distribution of a certain variable.
    Cuts can be indicated with lines and arrows. This class defines
    (and somehow enforces) a certain style.
    """

    def __init__(self, cuts=None,\
                 color_palette="dark",\
                 bins=None,\
                 xlabel=None):
        """
        Keyword Args:
            bins (array-like): desired binning, if None use default
            cuts (HErmes.selection.cut.Cut):
            color_palette (str): use this palette for the plotting
            xlabel (str): descriptive string for x-label
        """
        self.histograms = {}
        self.histratios = {}
        self.cumuls = {}
        self.plotratio = False
        self.plotcumul = False
        self.canvas = None
        if (xlabel is None):
            self.label = ''
        else:
            self.label = xlabel
        self.name = ''
        self.bins = bins
        if cuts is None:
            cuts = []
        self.cuts = cuts
        if isinstance(color_palette, str):
            self.color_palette = get_color_palette(color_palette)
        else:
            self.color_palette = color_palette
        self.plot_options = dict()

    def add_cuts(self, cut):
        """
        Add a cut to the the plot which can be indicated by an arrow

        Args:
            cuts (HErmes.selection.cuts.Cut):

        Returns:
            None
        """
        self.cuts.append(cut)

    def add_data(self, variable_data,\
                 name, bins=None,\
                 weights=None, label=''):
        """
        Histogram the added data and store internally
        
        Args:
            name (string): the name of a category
            variable_data (array): the actual data
        
        Keyword Args:
            bins (array): histogram binning
            weights (array): weights for the histogram
            label (str): A label for the data when plotted

        """
        if bins is None:
            bins = self.bins
        if weights is None:
            self.histograms[name] = d.factory.hist1d(variable_data, bins)
        else:
            Logger.debug(f"Found {len(weights)} weights and {len(variable_data)} data points")
            assert len(weights) == len(variable_data),\
                 f"Mismatch between len(weights) {len(weights)} and len(variable_data) {len(variable_data)}"
            self.histograms[name] = d.factory.hist1d(variable_data, bins, weights=weights)
        self.label = label
        self.name = name

    def add_variable(self, category,
                     variable_name,
                     external_weights=None,
                     transform=None):
        """
        Convenience interface if data is sorted in categories already

        Args:
            category (HErmese.variables.category.Category): Get variable from this category
            variable_name (string): The name of the variable

        Keyword Args:
            external_weights (np.ndarray): Supply an array for weighting. This will OVERIDE ANY INTERNAL WEIGHTING MECHANISM and use the supplied weights.
            transform (callable): Apply transformation todata

        """
        if category.plot_options:
            self.plot_options[category.name] = copy(category.plot_options)
        if self.bins is None:
            self.bins = category.vardict[variable_name].bins
        self.name = variable_name
        if external_weights is None:
            Logger.warning("Internal weighting mechanism is broken at the moment, FIXME!")
            #weights = category.weights
            #if len(weights) == 0:
            #    weights = None
            weights = None
        else:
            weights = external_weights
        if transform is None: transform = lambda x : x
        data = category.get(variable_name)
        #print (variable_name)
        #print (data)
        #print (data[0])
        # FIXME: check this
        # check pandas series and 
        # numpy array difference
        # FIMXME: values was as_matrix before - adapt changes in requirements.txt
        try:
            data = data.values
        except:
            pass
        # hack for applying the weights
        if hasattr(data[0],"__iter__"):
            if weights is not None:
                Logger.warning("Multi array data for {} detected. Trying to apply weights".format(variable_name))
                tmpweights = np.array([weights[i]*np.ones(len(data[i])) for i in range(len(data))])
                Logger.warning("Weights broken, assuming flatten as transformation")
                weights = flatten(tmpweights)

        self.add_data(transform(data),\
                      category.name,\
                      self.bins, weights=weights,\
                      label=category.vardict[variable_name].label)
        
    def add_cumul(self, name):
        """
        Add a cumulative distribution to the plot

        Args:
            name (str): the name of the category
        """
        assert name in self.histograms, "Need to add data first"
        self.cumuls[name] = self.histograms[name].normalized()

    def indicate_cut(self, ax, arrow=True):
        """
        If cuts are given, indicate them by lines

        Args:
            ax (pylab.axes): axes to draw on

        """
        vmin, vmax = ax.get_ylim()
        hmin, hmax = ax.get_xlim()
        for cut in self.cuts:
            for name, (operator, value) in cut:

                # there might be more than one
                # cut which should be
                # drawn on this plot
                # so go through ALL of them.
                if name != self.name:
                    continue

                Logger.debug('Found cut! {0} on {1}'.format(name,value))
                width = vmax/50.

                # create a line a cut position
                ax.vlines(value, ymin=vmin, ymax=vmax, linestyle=':')
                length = (hmax - hmin)*0.1

                # mark part of the plot as "forbidden"
                # and create arrow if desired
                if operator in ('>','>='):
                    shape = 'right'
                    ax.axvspan(value, hmax,\
                               facecolor=self.color_palette["prohibited"],\
                               alpha=0.5)
                else:
                    shape = 'left'
                    ax.axvspan(hmin, value,\
                               facecolor=self.color_palette["prohibited"],\
                               alpha=0.5)

                if arrow:
                    ax = create_arrow(ax, value, vmax*0.1,\
                                      -1., 0, length, width=width,\
                                      shape=shape, log=True)

    def add_ratio(self, nominator, denominator,\
                  total_ratio=None, total_ratio_errors=None, \
                  log=False, label="data/$\Sigma$ bg"):
        """
        Add a ratio plot to the canvas

        Args:
            nominator (list or str): name(s) of the categorie(s) which
                                     will be the nominator in the ratio
            denominator (list or str): name(s) of the categorie(s) which
                                     will be the nominator in the ratio

        Keyword Args:
            total_ratio (bool): Indicate the total ratio with a line in the plot
            total_ratio_errors (bool): Draw error region around total ratio
            log (bool): draw ratio plot in log-scale
            label (str): y-label for the ratio plot

        """
        if not isinstance(nominator, list):
            nominator = [nominator]
        if not isinstance(denominator, list):
            denominator = [denominator]

        name = "".join(nominator) + "_" + "".join(denominator)
        first_nominator = nominator.pop()
        nominator_hist = self.histograms[first_nominator]
        nominator_ws   = self.histograms[first_nominator].stats.weightsum
        for name in nominator:
            nominator_hist += self.histograms[name]
            nominator_ws   += self.histograms[name].stats.weightsum

        first_denominator = denominator.pop()
        denominator_hist = self.histograms[first_denominator]
        denominator_ws = self.histograms[first_denominator].stats.weightsum
        for name in denominator:
            denominator_hist += self.histograms[name]
            denominator_ws   += self.histograms[name].stats.weightsum
    
        nominator_hist.normalized()
        denominator_hist.normalized()
        ratio = d.histfuncs.histratio(nominator_hist, denominator_hist,\
                                      log=False, ylabel=label)
        if total_ratio is None:
            total_ratio = nominator_ws/denominator_ws
            Logger.info("Calculated scalar ratio of {:4.2f} from histos".format(total_ratio))

        #ratio.y[ratio.y > 0] = ratio.y[ratio.y > 0] + total_ratio -1
        self.histratios[name] = (ratio, total_ratio, total_ratio_errors,\
                                 label)
        return name

    def _draw_distribution(self, ax, name,
                                 log=True,\
                                 cumulative=False,
                                 normalized=False, 
                                 ylabel="rate/bin [1/s]"):
        """
        Paint the histograms!
        
        Args:    
    
        Keyword Args:
            normalized (bool): draw by number of events normalized distribution
                              

        """
        try:
            cfg = copy(self.plot_options[name])
        except KeyError:
            Logger.warning("No plot configuration available for {}".format(name))
            cfg = {"histotype": "line",
                   "label": name,
                   "linestyle" : {"color": "k",
                                  "linewidth": 3
                                  }
                   }
        log = False
        if "linestyle" in cfg: 
            color = cfg["linestyle"].pop('color')
            if isinstance(color, int):
                color = self.color_palette[color]
        if 'scatterstyle' in cfg:
            scattercolor = cfg["scatterstyle"].pop('color')

            if isinstance(scattercolor,int):
                scattercolor = self.color_palette[scattercolor]

        if cumulative:
            histograms = self.cumuls
            log = False
        else:
            histograms = self.histograms

        if normalized and not cumulative:
            histograms[name] = histograms[name].normalized()

        if cfg['histotype'] == 'scatter':
            histograms[name].scatter(log=log,cumulative=cumulative,\
                                     label=cfg["label"],\
                                     color=scattercolor, **cfg["scatterstyle"])
        elif cfg['histotype'] == "line":
            # apply th alpha only to the "fill" setting
            linecfg = copy(cfg["linestyle"])
            if "alpha" in linecfg:
                linecfg.pop("alpha")
                linecfg["filled"] = False
            if "filled" in cfg["linestyle"]:
                histograms[name].line(log=log, cumulative=cumulative,\
                                  label=cfg["label"], color=color,\
                                  **linecfg)

                histograms[name].line(log=log, cumulative=cumulative,\
                                  label=None, color=color,\
                                  **cfg["linestyle"])
            else:
                histograms[name].line(log=log, cumulative=cumulative,\
                                  label=cfg["label"], color=color,\
                                  **cfg["linestyle"])
        elif cfg['histotype'] == "overlay":
            histograms[name].line(log=log, cumulative=cumulative,\
                                  label=cfg["label"], color=color,\
                                  **cfg["linestyle"])
            histograms[name].scatter(log=log, cumulative=cumulative,\
                                     color=scattercolor,\
                                     **cfg["scatterstyle"])
        if cumulative:
            ax.set_ylabel('fraction')
        else:
            ax.set_ylabel(ylabel)

    def _draw_histratio(self, name, axes, ylim=(0.1,2.5)):
        """
        Plot one of the ratios
        
        Returns:
            tuple (float,float) : the min and max of the ratio
        """
        ratio,total_ratio,total_ratio_errors,label = self.histratios[name]
        ratio.scatter(c="k", marker="o", markersize=3)
        axes.hlines(total_ratio,axes.get_xlim()[0],axes.get_xlim()[1],linestyle="--")
        if total_ratio_errors is not None:
            axes.hlines(total_ratio + total_ratio_errors,axes.get_xlim()[0],axes.get_xlim()[1],linestyle=":")
            axes.hlines(total_ratio - total_ratio_errors,axes.get_xlim()[0],axes.get_xlim()[1],linestyle=":")
            xs = n.linspace(axes.get_xlim()[0],axes.get_xlim()[1],200)
            axes.fill_between(xs,total_ratio - total_ratio_errors,\
                              total_ratio + total_ratio_errors,\
                              facecolor="grey", alpha=0.3)
        #axes.set_ylim(ylim)
        axes.set_ylabel(label)
        axes.grid(1)
        if not ratio.y is None:
            if not ratio.yerr is None:
                ymin = min(ratio.y - ratio.yerr)
                ymax = max(ratio.y + ratio.yerr)
                ymin -= (0.1*ymin)
                ymax += (0.1*ymax)
                return ymin, ymax
            else:
                ymin = min(ratio.y)
                ymax = max(ration.y)
                ymin -= (0.1*ymin)
                ymax += (0.1*ymax)
                return ymin, ymax
        else:
            return 0,0

    def _locate_axes(self, combined_cumul, combined_ratio, combined_distro):
        axes_locator = []
        if self.cumuls:
            if combined_cumul:
                axes_locator.append((0, "c"))
            else:
                axes_locator += [(x,"c") for x in range(len(self.cumuls))]

        if self.histratios:
            if combined_ratio:
                if axes_locator:
                    axes_locator.append((axes_locator[-1] + 1,"r"))
                else:
                    axes_locator.append((0,"r"))
            else:
                if axes_locator:
                    axes_locator +=[(x+ len(axes_locator),"r") for x  in range(len(self.histratios))]

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

    def plot(self,
             axes_locator=((0, "c",.2), (1, "r",.2), (2, "h",.5)),\
             combined_distro=True,\
             combined_ratio=True,\
             combined_cumul=True,
             normalized=True,
             style="classic",\
             log=True,
             legendwidth = 1.5,
             ylabel="rate/bin [1/s]",
             figure_factory=None,
             zoomin=False,
             adjust_ticks=lambda x : x):
        """
        Create the plot

        Keyword Args:
            axes_locator (tuple): A specialized tuple defining where the axes should be located in the plot
                                  tuple has the following form: 
                                  ( (PLOTA), (PLOTB), ...) where PLOTA is a tuple itself of the form (int, str, int)
                                  describing (plotnumber, plottype, height of the axes in the figure)
                                  plottype can be either: "c" - cumulative
                                                          "r" - ratio
                                                          "h" - histogram
            combined_distro:
            combined_ratio:
            combined_cumul:
            log (bool):
            style (str): Apply a simple style to the plot. Options are "modern" or "classic"
            normalized (bool):
            figure_factor (fcn): Must return a matplotlib figure, use for custom formatting
            zoomin (bool): If True, select the yrange in a way that the interesting part of the 
                           histogram is shown. Caution is needed, since this might lead to an
                           overinterpretation of fluctuations.
            adjust_ticks (fcn): A function, applied on a matplotlib axes
                                which will set the proper axis ticks


        Returns:

        """

        Logger.info("Found {} distributions".format(len(self.histograms)))
        Logger.info("Found {} ratios".format(len(self.histratios)))
        Logger.info("Found {} cumulative distributions".format(len(self.cumuls)))
        if not axes_locator:
            axes_locator = self._locate_axes(combined_cumul,\
                                             combined_ratio,\
                                             combined_distro)

        # calculate the amount of needed axes
        # assert len(axes_locator) == len(heights), "Need to specify exactly as many heights as plots you want to have"

        heights = [k[2] for k in axes_locator]
        self.canvas = YStackedCanvas(subplot_yheights=heights,\
                                     figure_factory=figure_factory)
        
        cu_axes = [x for x in axes_locator if x[1] == "c"]
        h_axes = [x for x in axes_locator if x[1] == "h"]
        r_axes = [x for x in axes_locator if x[1] == "r"]

        maxheights = []
        minheights = []
        for ax in cu_axes:
            cur_ax = self.canvas.select_axes(ax[0])
            if combined_cumul:
                for k in list(self.cumuls.keys()):
                    self._draw_distribution(cur_ax, k, cumulative=True,log=log, ylabel=ylabel)
                break
            else:
                k = self.cumuls[list(self.cumuls.keys())[ax[0]]]
                self._draw_distribution(cur_ax,cumulative=True,log=log, ylabel=ylabel)


        for ax in r_axes:
            cur_ax = self.canvas.select_axes(ax[0])
            if combined_ratio:
                for k in list(self.histratios.keys()):
                    ymin, ymax = self._draw_histratio(k,cur_ax)
                    ymin -= (ymin*0.1)
                    ymax += (ymax*0.1)
                    cur_ax.set_ylim(ymin,ymax)

                    # FIXME: good tick spacing
                    #major_tick_space = 1
                    #minor_tick_space = 0.1
                    ## in case there are too many ticks
                    #nticks = float(ymax - ymin)/major_tick_space
                    #while nticks > 4:
                    #    major_tick_space += 1 

                    ##if ymax - ymin < 1:
                    ##    major_tick_space =  
                    #cur_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(minor_tick_space))
                break
            else:
                k = self.histratios[list(self.histratios.keys())[ax[0]]]
                ymin, ymax = self._draw_histratio(k,cur_ax)    
                ymin -= (ymin*0.1)
                ymax += (ymax*0.1)
                cur_ax.set_ylim(ymin,ymax)

        for ax in h_axes:
            cur_ax = self.canvas.select_axes(ax[0])
            if combined_distro:
                for k in list(self.histograms.keys()):
                    Logger.debug("drawing..{}".format(k))
                    self._draw_distribution(cur_ax,k,log=log, normalized=normalized, ylabel=ylabel)
                break
            else:
                k = self.histograms[list(self.histograms.keys())[ax[0]]]
                ymax, ymin = self._draw_distribution(cur_ax,k,log=log, normalized=normalized, ylabel=ylabel)
            cur_ax.set_ylim(ymin=ymin - 0.1*ymin,ymax=1.1*ymax)
            cur_ax.grid(True)
        lgax = self.canvas.select_axes(-1) # most upper one
        ncol = 2 if len(self.histograms) <= 4 else 3 
        if style == "classic":
            # draw the legend in the box above the plot
            legend_kwargs = {"bbox_to_anchor": [0., 1.0, 1., .102],
                             "loc": 3,
                             "frameon": True,
                             "ncol": ncol,
                             "framealpha": 1.,
                             "borderaxespad": 0,
                             "mode": "expand",
                             "handlelength": 2,
                             "numpoints": 1}
            lg = lgax.legend(**legend_kwargs)
            if lg is not None:
                lg.get_frame().set_linewidth(legendwidth)
                lg.get_frame().set_edgecolor("k")
            else:
                Logger.warning("Can not set legendwidth!")       
        if style == "modern":
            # be more casual
            lgax.legend() 

        # plot the cuts
        if self.cuts:
            for ax in h_axes:
                cur_ax = self.canvas.select_axes(ax[0])
                self.indicate_cut(cur_ax, arrow=True)
            for ax in r_axes + cu_axes:
                cur_ax = self.canvas.select_axes(ax[0])
                self.indicate_cut(cur_ax, arrow=False)
        # cleanup
        leftplotedge, rightplotedge, minplotrange, maxplotrange = self.optimal_plotrange_histo(self.histograms.values())
        if minplotrange == maxplotrange:
            Logger.debug("Detected histogram with most likely a single bin!")
            Logger.debug("Adjusting plotrange")
            
        else:
            if zoomin: 
                figure_span = maxplotrange - minplotrange
                minplotrange -= (figure_span*0.1)
                maxplotrange += (figure_span*0.1) 
            else: # start at zero and show the boring part
                minplotrange = 0
                maxplotrange += (maxplotrange*0.1)
        if log:
            maxplotrange = 10**(np.log10(maxplotrange) + 1)

            if maxplotrange < 1: 
                minplotrange -= (minplotrange*0.01)
            else:
                minplotrange = 0 # will be switched to symlog by default

        for ax in h_axes:
            self.canvas.select_axes(ax[0]).set_ylim(ymin=minplotrange, ymax=maxplotrange)
            self.canvas.select_axes(ax[0]).set_xlim(xmin=leftplotedge, xmax=rightplotedge)
            if log:
                if maxplotrange < 1:
                    self.canvas.select_axes(ax[0]).set_yscale("log")
                else:
                    self.canvas.select_axes(ax[0]).set_yscale("symlog")

        for ax in cu_axes:
            self.canvas.select_axes(ax[0]).set_xlim(xmin=leftplotedge, xmax=rightplotedge)
        for ax in r_axes:
            self.canvas.select_axes(ax[0]).set_xlim(xmin=leftplotedge, xmax=rightplotedge)
        self.canvas.eliminate_lower_yticks()
        # set the label on the lowest axes
        self.canvas.axes[0].set_xlabel(self.label)
        minor_tick_space = self.canvas.axes[0].xaxis.get_ticklocs()
        minor_tick_space = (minor_tick_space[1] - minor_tick_space[0])/10.
        if minor_tick_space < 0.1:
            Logger.debug("Adjusting for small numbers in tick spacing, tickspace detectected {}".format(minor_tick_space))
            minor_tick_space = 0.1
        self.canvas.axes[0].xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(minor_tick_space))
        for x in self.canvas.axes[1:]:
            p.setp(x.get_xticklabels(), visible=False)
            p.setp(x.get_xticklines(), visible=False)
            x.xaxis.set_tick_params(which="both",\
                                    length=0,\
                                    width=0,\
                                    bottom=False,\
                                    labelbottom=False)
        for x in self.canvas.figure.axes:
            x.spines["right"].set_visible(True)
            adjust_ticks(x)

        for ax in h_axes:
            #self.canvas.select_axes(ax[0]).ticklabel_format(useOffset=False, style='plain', axis="y")
            self.canvas.select_axes(ax[0]).get_yaxis().get_offset_text().set_x(-0.1)

        if ((len(h_axes) == 1) and (style == "modern")):
            self.canvas.select_axes(-1).spines["top"].set_visible(False)
            self.canvas.select_axes(-1).spines["right"].set_visible(False)


    @staticmethod
    def optimal_plotrange_histo(histograms):
        """
        Get most suitable x and y limits for a bunc of histograms
        
        Args:
            histograms (list(d.factory.hist1d)): The histograms in question

        Returns:
            tuple (float, float, float, float): xmin, xmax, ymin, ymax

        """

        leftplotedge = n.inf
        rightplotedge = -n.inf
        minplotrange = n.inf
        maxplotrange = -n.inf
        for h in histograms:
            if not h.bincontent.any():
                continue
            if h.bincenters[h.bincontent > 0][0] < leftplotedge:
                leftplotedge = h.bincenters[h.bincontent > 0][0]
                leftplotedge -= h.binwidths[0]

            if h.bincenters[h.bincontent > 0][-1] > rightplotedge:
                rightplotedge = h.bincenters[h.bincontent > 0][-1]
                rightplotedge += h.binwidths[0] 

            if min(h.bincontent[h.bincontent > 0]) < minplotrange:
                minplotrange = min(h.bincontent[h.bincontent > 0])
                
            if max(h.bincontent[h.bincontent > 0]) > maxplotrange:
                maxplotrange = max(h.bincontent[h.bincontent > 0])

        Logger.info("Estimated plotrange of xmin {} , xmax {}, ymin {}, ymax {}".format(leftplotedge, rightplotedge, minplotrange, maxplotrange)) 
        return leftplotedge, rightplotedge, minplotrange, maxplotrange


    def add_legend(self, **kwargs):
        """
        Add a legend to the plot. If no kwargs are passed,
        use some reasonable default.
        
        Keyword Args:
             will be passed to pylab.legend
        """
        if not kwargs:
            kwargs = {"bbox_to_anchor": (0.,1.0, 1., .102),\
                      "loc" : 3, "ncol" :3,\
                      "mode": "expand",\
                      "framealpha": 1,\
                      "borderaxespad": 0.,\
                      "handlelength": 2,\
                      "numpoints": 1}
        self.canvas.global_legend(**kwargs)
 
# #######################################################
#
# def error_distribution_plot(h,
#                             xlabel = r"$\log(E_{rec}/E_{ref})$",
#                             name = "E",
#                             median = False):
#     """
#
#
#     Args:
#         h:
#         xlabel:
#         name:
#         median:
#
#     Returns:
#
#     """
#     par = HistoFitter(h, Gauss)
#     fig  = p.figure(figsize=(6,4),dpi=350)
#     ax   = fig.gca()
#     if not median: ax.plot(h.bincenters, Gauss(par,h.bincenters),color="k",lw=2)
#     h.line(filled=True,color="k",lw=2,fc="grey",alpha=.5)#hatch="//")
#     h.line(color="k",lw=2)
#     ax.grid(1)
#     ax.set_ylim(ymax=1.1*max(h.bincontent))
#     ax.set_xlim(xmin=h.bincenters[0],xmax=h.bincenters[-1])
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel("Normalized bincount")
#     if median: ax.vlines(h.stats.median,0,1.1*max(h.bincontent),linestyles="dashed")
#     textstr ="Gaussian fit:\n"
#     textstr += "$\mu$ = " + "%4.3f" %par[1] + "\n" + "$\sigma$ = " + "%4.2f" %par[2]
#     if median:
#         textstr = "Median:\n %4.3f" %h.stats.median
#     CreateTextbox(ax,textstr,boxstyle="square",xcoord=.65,fontsize=16,alpha=.9)
#     #Thesisize(ax)
#     #print h.bincontent[h.bincenters > -.1][h.bincenters < .1].cumsum()[-1]
#
#     #ChisquareTest(h,Gauss(par,h.bincenters),xmin=-.001,xmax=.001)
#     #savename = Multisavefig(plotdir_stat,"parameter-reso-" + name,3,orientation="portrait",pad_inches=.3,bbox_inches="tight")[0]#pad_inches=.5,bbox_inche     s="tight")[0]
#     return fig
#
# ####################################################
#
# def HistoFitter(histo,func,startmean=0,startsigma=.2):
#
#     def error(p,x,y):
#         return n.sqrt((func(p,x) - y)**2)
#
#     #print histo.bincontent.std()
#     histo.stats.mean
#     p0 = [max(histo.bincontent),histo.stats.mean,histo.stats.var]
#     output = optimize.leastsq(error,p0,args=(histo.bincenters,histo.bincontent),full_output=1)
#     par = output[0]
#     covar = output[1]
#     rchisquare = scipy.stats.chisquare(1*histo.bincontent,f_exp=(1*func(par,histo.bincenters)))[0]/(1*(len(histo.bincenters) -len(par)))
#     #print par,covar
#     #print "chisquare/ndof",rchisquare
#     #print histo.bincontent[:10], func(par,histo.bincenters)[:10]
#     #print "ks2_samp", scipy.stats.ks_2samp(histo.bincontent,func(par,histo.bincenters))
#     return par
#
# #####################################################
#
# def create_textbox(ax, textstr, boxstyle="round",\
#                    facecolor="white", alpha=.7,\
#                    xcoord=0.05, ycoord=0.95, fontsize=14):
#     """
#     Create a textbox on a given axis
#
#     Args:
#         ax:
#         textstr:
#         boxstyle:
#         facecolor:
#         alpha:
#         xcoord:
#         ycoord:
#         fontsize:
#
#     Returns:
#         the given ax object
#     """
#     props = dict(boxstyle=boxstyle, facecolor=facecolor, alpha=alpha)
#     # place a text box in upper left in axes coords
#     ax.text(xcoord, ycoord, textstr,\
#             transform=ax.transAxes,\
#             fontsize=fontsize,\
#             verticalalignment='top', bbox=props)
#     return ax
#
# ######################################################
#
# def ChisquareTest(histo, fit, xmin=-.2, xmax=.2, ndof=3):
#     data = histo.bincontent[histo.bincenters > xmin][histo.bincenters < xmax]
#     fit  = fit[histo.bincenters > xmin][histo.bincenters < xmax]
#     #print data,fit
#     print (scipy.stats.chisquare(data,f_exp=fit)[0]/(len(fit) - ndof))
#     print (scipy.stats.ks_2samp(data,fit))
#     print (scipy.stats.anderson(data))


