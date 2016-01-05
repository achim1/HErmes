

import os.path
import dashi as d
import pylab as p
import yaml

#
from pyevsel.utils.logger import Logger
from IPython.core.display import Image

STD_CONF=os.path.join(os.path.split(__file__)[0],"plotsconfig.yaml")


# golden cut values
# CW = current width of my thesis (adjust
CW = 5.78851
S  = 1.681



def GetCanvasConfig(filename=STD_CONF):
    data = yaml.load(open(filename,"r"))
    return data["canvas"]

def GetSavefigConfig(filename=STD_CONF):
    data = yaml.load(open(filename,"r"))
    return data["savefig"]


#pylab axes predefinition
#p.axes([left,bottom,width,height]..

##########################################

class YStackedCanvas(object):
    """
    A canvas for plotting multiple axes
    """

    def __init__(self,axeslayout=[.2,.2,.5],figsize=(CW,CW*S)):
        """
        Axes indices go from bottom to top
        """

        self.figure = p.figure(figsize=figsize)
        self._create_top_stacked_axes(heights=axeslayout)
        self.png_filename = None
        for subplot in self.axes:
            self.figure.add_axes(subplot)
            
        self.figure.subplots_adjust(hspace=0)

    def _create_top_stacked_axes(self,heights=[1.],configfile=STD_CONF):
        """
        heights is relative, e.g.
        heights = [.2,.1,.6] will give axes using this amount of 
        space
        """
        assert sum(heights) <= 1., "heights must be in relative heights"
    
        cfg = GetCanvasConfig()
        left = cfg["leftpadding"]
        right = cfg["rightpadding"]
        bot  = cfg["bottompadding"]
        top  = cfg["toppadding"]
        width = 1. - left - right
        height = 1. - top - bot
        
        heights = [height*h for h in heights]
        abs_bot = 0. + bot
    
        axes = [p.axes([left,abs_bot,width,heights[0]])]
        for h in heights[1:]:
            theaxes = p.axes([left,abs_bot,width,h])
            p.setp(theaxes.get_xticklabels(), visible=False)
            axes.append(theaxes)
            abs_bot += h
    
        self.axes = axes


    def select_axes(self, axes):
        ax = self.axes[axes]
        p.sca(ax)
        return ax

    def save(self,path,name,endings=["pdf","png"],**kwargs):
        """
        Calls pylab.savefig for all endings
        """
        if not kwargs:
            kwargs = GetSavefigConfig()

        for ending in endings:
            filename = os.path.join(path,name + '.' + ending)
            self.figure.savefig(filename,format=ending,**kwargs)
            Logger.info("Saved to %s" %filename)
            if ending == "png":
                self.png_filename = filename
        return self.png_filename

    def show(self):
        return Image(self.png_filename)

######################################################
#
#class _AbstractMultiPlotBase(object):
#    """
#    A base class for all MultiPlots
#    """
#    
#    def __init__(self,figsize=(6,8),dpi=450):
#        self.ax             = dict()
#        self.dpi            = dpi
#        self.all_stats      = dict()
#        self.figure         = p.figure(figsize=figsize,dpi=dpi)
#        self.savefig_kwargs = dict()
#        self.legend         = None 
#        self.primary_axes   = None
#        self.png_savepath   = ""
#
#    def select_axes(self, axes):
#        ax = self.ax[axes]
#        p.sca(ax)
#        return ax
#
#    def global_legend(self, legitems=None, **kwargs):
#        if legitems is None:
#            handles, labels = self.primary_axes.get_legend_handles_labels()
#            self.legend = p.legend(**kwargs)#bbox_to_anchor=self.leg_bbox,
#                #bbox_transform=self.figure.transFigure,
#                #mode="expand", borderaxespad=0,
#                #loc="lower left",
#                #**kwargs)
#        else:
#            handles, labels = self.primary_axes.get_legend_handles_labels()
#            self.legend = p.legend([i[0] for i in legitems],
#                     [i[1] for i in legitems],**kwargs)
#            #        bbox_to_anchor=self.leg_bbox,
#            #        bbox_transform=self.figure.transFigure,
#            #        mode="expand", borderaxespad=0,
#            #        loc="lower left",
#            #        **kwargs)
#    def get_axnames(self):
#        return self.ax.keys()
#    
#    def save(self,plotdir,name,savefiles=1,savefig_kwargs={}):
#        if savefig_kwargs: # change the settings on the fly
#            self.savefig_kwargs = savefig_kwargs
#
#        name  = name.replace("_","-")
#        #self.savefig_kwargs["bbox_extra_artists"] = (self.legend,)
#        self.savefig_kwargs['bbox_inches'] = "tight"
#        savenames = []
#        for ending in enumerate(["png","pdf","eps"]):
#            if ending[0] < savefiles:
#                self.savefig_kwargs["format"] = ending[1]
#                savefig(join(plotdir,name + "." + ending[1]),**self.savefig_kwargs)
#                savenames.append(join(plotdir,name + "." + ending[1]))
#        Logger.info("Saving %s " %savenames.__repr__())
#        self.png_savepath = savenames[0]
#        return savenames[0]
#        
#        #savename = Multisavefig(plotdir,name, savefiles=savefiles)
#        #if docs:
#            #doctext = self.__doc__
#            #if doctext is None:
#            #    doctext = ""
#            #self.docs = FigureDocumentation(figure=join(plotdir,name),title=name,filename=name + ".rst",directory='/afs/ifh.de/user/s/stoessl/doc_src/cascades/source/plots',ana_level = self.ana_level,text = doctext )
#            ##self.docs.addText(self.__doc__)
#            #or key in self.all_stats.keys():
#            #    self.docs.addSimpleTable(self.all_stats[key], key)
#            
#            #self.docs.save()
#
#
#class MultiPlot(_AbstractMultiPlotBase):
#    """
#    A matplotlib canvas with several panels
#    """
#    all_stats = {}
#    
#    def __init__(self,equalsize=False,panels=None,ana_level="",\
#                 figsize=(8,9),dpi=450,leftpadding=0.15, topextrapadding=0.,\
#                 transparent=False,facecolor="w",edgecolor="w",\
#                 bbox_inches="tight"):
#        #_AbstractMultiPlotBase.__init__(self,figsize=figsize,dpi=dpi)
#        self.ax             = dict()
#        self.dpi            = dpi
#        self.all_stats      = dict()
#        self.figure         = p.figure(figsize=figsize,dpi=dpi)
#        self.savefig_kwargs = dict()
#        self.legend         = None
#        self.primary_axes   = None
#        self.png_savepath   = ""
#        self.figsize   = figsize
#        self.ax = dict()
#        self.savefig_kwargs = {"dpi" : dpi,"transparent" : transparent,\
#                               "facecolor" : facecolor,"edgecolor" : edgecolor,\
#                               "bbox_inches" : bbox_inches}
#
#        self.ana_level = ana_level
#
#        self.leg_bbox  = (.15, .855, .8, .1)
#
#        
#        if panels <= 2:
#            equalsize = True
#        #equalsize   = (panels <= 2)    
# 
#        if equalsize:
#            if panels   == 1:
#                self.ax["center"]   = p.axes([leftpadding,.10,.8,.80-topextrapadding])
#                self.primary_axes = self.ax["center"]
#            elif panels == 2:
#                self.ax["upper"]    = p.axes([leftpadding,.50,.8,.40-topextrapadding/2])
#                self.ax["lower"]    = p.axes([leftpadding,.10,.8,.40-topextrapadding/2], sharex=self.ax["upper"])
#                self.primary_axes = self.ax["upper"]
#                                
#            elif panels == 4:
#                self.ax["upper"]    = p.axes([leftpadding,.70,.8,.20-topextrapadding/4])
#                self.ax["center"]   = p.axes([leftpadding,.50,.8,.20-topextrapadding/4], sharex=self.ax["upper"])
#                self.ax["lower"]    = p.axes([leftpadding,.30,.8,.20-topextrapadding/4], sharex=self.ax["center"])
#                self.ax["bottom"]   = p.axes([leftpadding,.10,.8,.20-topextrapadding/4], sharex=self.ax["lower"])
#                self.primary_axes = self.ax["upper"]
#            else:
#                self.ax["upper"]    = p.axes([leftpadding,.60,.8,.25-topextrapadding/3])
#                self.ax["center"]   = p.axes([leftpadding,.35,.8,.25-topextrapadding/3], sharex=self.ax["upper"])
#                self.ax["lower"]    = p.axes([leftpadding,.10,.8,.25-topextrapadding/3], sharex=self.ax["center"])
#                self.primary_axes = self.ax["upper"]
#                
#        
#            
#        else:
#            #self.ax["lower"]    = p.axes([.15,.10,.8,.20])#, sharex=self.ax["center"])
#            self.ax["upper"]    = p.axes([.15,.50,.8,.35])#, sharex=self.ax["lower"])
#            self.ax["center"]   = p.axes([.15,.30,.8,.20], sharex=self.ax["upper"])
#            self.ax["lower"]    = p.axes([.15,.10,.8,.20], sharex=self.ax["center"])
#            self.primary_axes = self.ax["upper"]
#
#        for subplot in self.ax.keys():
#            self.figure.add_axes(self.ax[subplot])
#            
#        self.figure.subplots_adjust(hspace=0)
#        
#        # check xticklables 
#        if panels == 2:
#            p.setp(self.ax["upper"].get_xticklabels(), visible=False)
#        if panels > 2:
#            p.setp(self.ax["upper"].get_xticklabels(), visible=False)
#            p.setp(self.ax["center"].get_xticklabels(), visible=False)
#        if panels > 3:
#            p.setp(self.ax["lower"].get_xticklabels(), visible=False)
#
################################################
#
#class NPanelView(_AbstractMultiPlotBase):
#    
#    
#    def __init__(self,nplots,ncols,ana_level="3a",figsize=(8,9),dpi=450):
#        _AbstractMultiPlotBase.__init__(self,figsize=figsize,dpi=dpi)
#        self.ana_level = ana_level
#        nrows = nplots/ncols
#        if nplots%ncols:
#            nrows += 1
#            
#        fig, axes = p.subplots(nrows=nrows,ncols=ncols, sharex=True,sharey=True,figsize=figsize,dpi=dpi)
#        self.ax = dict()
#        for i in range(nrows):
#            for j in range(ncols):
#                axes[i][j].grid(1)
#                self.ax["ax_%i%i" %(i,j)] = axes[i][j]
#                
############################################
#
#class FourPanelView(_AbstractMultiPlotBase):
#    
#    def __init__(self,ana_level="3a",figsize=(8,9),dpi=450):
#        _AbstractMultiPlotBase.__init__(self,figsize=figsize,dpi=dpi)
#        self.ana_level = ana_level
#        fig, axes = p.subplots(nrows=2,ncols=2, sharex=True,sharey=True,figsize=figsize,dpi=dpi)
#
#        for i in range(2):
#            for j in range(2):
#                axes[i][j].grid(1)
#                
#        self.ax = dict()
#        self.ax["upper_left"]  = axes[0][0]
#        self.ax["upper_right"] = axes[0][1]
#        self.ax["lower_left"]  = axes[1][0]
#        self.ax["lower_right"] = axes[1][1]
#
#         
#        
#class VariablePlot(MultiPlot):
#
#    def __init__(self,panels=None):
#        Logger.warning("This will go away soon!")
#        super(VariablePlot,self).__init__(panels=panels)
#
#    
#
#
#if __name__ == "__main__":
#    
#    
#    import numpy as n
#    x = MultiPlot()
#    x.set_nbins(n.linspace(0,5,100))
#    p.show()
