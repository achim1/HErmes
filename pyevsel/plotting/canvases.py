"""
Provides canvases for multi axes plots
"""

from builtins import object
import os.path
import pylab as p
import tempfile

from pyevsel.utils.logger import Logger
try:
    from IPython.core.display import Image
except ImportError:
    try:
        from IPython.display import DisplayObject as Image
    except ImportError:
        Logger.debug("Can not import IPython!")
        def Image(x): x

# golden cut values
from .layout import FIGSIZE_A4_SQUARE, FIGSIZE_A4, FIGSIZE_A4_LANDSCAPE

##########################################


class YStackedCanvas(object):
    """
    A canvas for plotting multiple axes
    """

    def __init__(self, subplot_yheights=(.2,.2,.5),\
                       padding=(0.15, 0.05, 0.0, 0.1 ),\
                       figsize="auto"):
        """
        Create a new canvas with multiple axes on top of each other.
        The height of the individual axes can be specified.
        In most cases, the defaults should be reasonable.

        Keyword Args:
            subplot_yheights (iterable): The normalized heights of the individual
                                         subplots. Sorted from bottom to top
            padding (left, right, top, bottom): The padding around the figure
            figsize (width, height) or str: The size of the figure (in inches). Keyword "auto" 
                                            means the plot figures it out by itself.
        """
        assert sum(subplot_yheights) <= 1., "subplot_yheights must be in relative heights"
        assert len(padding) == 4, "Needs 4 values for padding (left, right, top, bottom)"

        self.axes = None
        if figsize == "auto": # FIXME: something more clever should happen here
            self.figure = p.figure(figsize=FIGSIZE_A4_SQUARE)
        else:
            self.figure = p.figure(figsize=figsize)

        # self.create_top_stacked_axes(heights=axeslayout)
        left, right, top, bot = padding
        width = 1. - left - right
        height = 1. - top - bot

        # calculate absolute heights and stack the axes
        # from bottom to top
        heights = [height * h for h in subplot_yheights]
        heights.reverse()
        Logger.debug("Using heights {0}".format(heights.__repr__()))
        abs_bot = 0 + bot
        axes = [p.axes([left, abs_bot, width, heights[0]])]
        abs_bot = bot + heights[0]
        for h in heights[1:]:
            theaxes = p.axes([left, abs_bot, width, h])
            p.setp(theaxes.get_xticklabels(), visible=False)
            axes.append(theaxes)
            abs_bot += h

        self.axes = axes
        self.png_filename = None
        for subplot in self.axes:
            self.figure.add_axes(subplot)

        self.figure.subplots_adjust(hspace=0)
        self.savekwargs = dict()


    # def create_top_stacked_axes(self, heights=(1.)):
    #     """
    #     Create several axes for subplot on top of each other
    #
    #     Args:
    #         heights (iterable):  relative height e.g.
    #                              heights = [.2,.1,.6] will give axes using this amount of
    #                              space
    #     """
    #
    #     cfg = get_config_item("canvas")
    #     left = cfg["leftpadding"]
    #     right = cfg["rightpadding"]
    #     bot  = cfg["bottompadding"]
    #     top  = cfg["toppadding"]
    #     width = 1. - left - right
    #     height = 1. - top - bot
    #
    #     heights = [height*h for h in heights]
    #     heights.reverse()
    #     Logger.debug("Using heights {0}".format(heights.__repr__()))
    #     abs_bot = 0 + bot
    #     axes = [p.axes([left, abs_bot,width,heights[0]])]
    #     restheights = heights[1:]
    #     abs_bot = bot + heights[0]
    #     for h in restheights:
    #         theaxes = p.axes([left,abs_bot,width,h])
    #         p.setp(theaxes.get_xticklabels(), visible=False)
    #         axes.append(theaxes)
    #         abs_bot += h
    #
    #     self.axes = axes

    def limit_yrange(self, ymin=None, ymax=None):
        """
        Walk through all axes and adjust ymin and ymax

        Keyword Args:
            ymin (float): min ymin value
        """
        for ax in self.axes:
            if ymin is not None:
                ax.set_ylim(ymin=ymin)
            if ymax is not None:
                ax.set_ylim(ymax=ymax)

    def limit_xrange(self, xmin=None, xmax=None):
        """
        Walk through all axes and set xlims

        Keyword Args:
            xmin (float): left x edge of axes
            xmax (float): right x edge of axes

        Returns:
            None
        """
        for ax in self.axes:
            if xmin is not None:
                ax.set_xlim(xmin=xmin)

            if xmax is not None:
                ax.set_xlim(xmax=xmax)

    def eliminate_lower_yticks(self):
        """
        Eliminate the lowest y tick on each axes.
        The bottom axes keeps its lowest y-tick.
        """    
        for ax in self.axes[1:]:
            ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
            ax.yaxis.get_major_ticks()[-1].label1.set_visible(False)

    def select_axes(self, axes):
        """
        Set the scope on a certain axes

        Args:
            axes (int): 0, lowest, -1 highest, increasing y-order

        Returns:
            matplotlib.axes.axes: The axes intance
        """
        ax = self.axes[axes]
        p.sca(ax)
        return ax

    def global_legend(self, *args, **kwargs):
        """
        A combined legend for all axes

        Args:
            all args will be passed to pylab.legend

        Keyword Args:
            all kwargs will be passed to pylab.legend

        """
        handles, labels = self.select_axes(-1).get_legend_handles_labels()
        if args:
            args = [i[1] for i in args]
        self.legend = p.legend(*args, **kwargs)
        #else:
        #    self.legend = p.legend([i[0] for i in legitems],
        #             [i[1] for i in legitems],**kwargs)
            #        bbox_to_anchor=self.leg_bbox,
            #        bbox_transform=self.figure.transFigure,
            #        mode="expand", borderaxespad=0,
            #        loc="lower left",
            #        **kwargs)

    def save(self, path, name, formats=("pdf", "png"), **kwargs):
        """
        Calls pylab.savefig for all endings

        Args:
            path (str): path to savefile
            name (str): filename to save
            formats (tuple): for each name in endings, a file is save

        Keyword Args:
            all keyword args will be passed to pylab.savefig

        Returns:
            str: The full path to the the saved file
        """

        ending = name.split(".")[-1]
        if ending in formats:
            name = name.replace("." + ending, "")
        for theformat in formats:
            filename = os.path.join(path, name + '.' + theformat)
            self.figure.savefig(filename, format=theformat, **kwargs)
            if theformat == "png":
                self.png_filename = filename
        return self.png_filename

    def show(self):
        """
        Use the IPython.core.Image to show the plot

        Returns:
            IPython.core.Image: the plot
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            self.figure.savefig(f, format="png")
            return Image(f.name)


