"""
Provides canvases for multi axes plots
"""

import os.path
import pylab as p

from pyevsel.plotting import get_config_item
from pyevsel.utils.logger import Logger
try:
    from IPython.core.display import Image
except ImportError:
    Logger.debug("Can not import IPython!")
    Image = lambda x : x

# golden cut values
# CW = current width of my thesis (adjust
CW = 5.78851
S  = 1.681

##########################################

class YStackedCanvas(object):
    """
    A canvas for plotting multiple axes
    """

    def __init__(self,axeslayout=(.2,.2,.5),figsize=(CW,CW*S)):
        """
        Axes indices go from bottom to top
        """

        self.figure = p.figure(figsize=figsize)
        self.create_top_stacked_axes(heights=axeslayout)
        self.png_filename = None
        for subplot in self.axes:
            self.figure.add_axes(subplot)
            
        self.figure.subplots_adjust(hspace=0)
        self.savekwargs = dict()

    def create_top_stacked_axes(self,heights=(1.)):
        """
        Create several axes for subplot on top of each other

        Args:
            heights (iterable):  relative height e.g.
                                 heights = [.2,.1,.6] will give axes using this amount of
                                 space
        """
        assert sum(heights) <= 1., "heights must be in relative heights"
    
        cfg = get_config_item("canvas")
        left = cfg["leftpadding"]
        right = cfg["rightpadding"]
        bot  = cfg["bottompadding"]
        top  = cfg["toppadding"]
        width = 1. - left - right
        height = 1. - top - bot
        
        heights = [height*h for h in heights]
        heights.reverse()
        Logger.debug("Using heights {0}".format(heights.__repr__()))
        abs_bot = 0 +bot     
        axes = [p.axes([left,abs_bot,width,heights[0]])]
        restheights = heights[1:]
        abs_bot = bot + heights[0]
        for h in restheights:
            theaxes = p.axes([left,abs_bot,width,h])
            p.setp(theaxes.get_xticklabels(), visible=False)
            axes.append(theaxes)
            abs_bot += h
    
        self.axes = axes

    def limit_yrange(self,ymin=10**(-12)):
        """
        Walk through all axes and adjust ymin

        Keyword Args:
            ymin (float): min ymin value
        """
        for ax in self.axes:
            p.sca(ax)
            axymin,__ =  ax.get_ylim()    
            if abs(axymin) < ymin:
                ax.set_ylim(ymin=ymin)

    def eliminate_lower_yticks(self):
        """
        Eliminate the lowest y tick on each axes 
        which is not the lowest
        """    
        for ax in self.axes[1:]:
            #p.sca(ax)
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
        self.legend = p.legend(*args,**kwargs)
        #else:
        #    self.legend = p.legend([i[0] for i in legitems],
        #             [i[1] for i in legitems],**kwargs)
            #        bbox_to_anchor=self.leg_bbox,
            #        bbox_transform=self.figure.transFigure,
            #        mode="expand", borderaxespad=0,
            #        loc="lower left",
            #        **kwargs)

    def save(self,path,name,endings=("pdf","png"),**kwargs):
        """
        Calls pylab.savefig for all endings

        Args:
            path (str): path to savefile
            name (str): filename to save
            endings (tuple): for each name in endings, a file is save

        Keyword Args:
            all keyword args will be passed to pylab.savefig

        Returns:
            str: The full path to the the saved file
        """
        if not kwargs:
            kwargs = get_config_item("savefig")

        self.savekwargs = kwargs
        self.savekwargs.update({'path' : path,'name' : name, 'endings' : endings})
        for ending in endings:
            filename = os.path.join(path,name + '.' + ending)
            self.figure.savefig(filename,format=ending,**kwargs)
            if ending == "png":
                self.png_filename = filename
        return self.png_filename

    def show(self):
        """
        Use the IPython.core.Image to show the plot

        Returns:
            IPython.core.Image: the plot
        """
        path = self.savekwargs.pop('path')
        name = self.savekwargs.pop('name')
        self.save(path,name,**self.savekwargs)
        return Image(self.png_filename)


