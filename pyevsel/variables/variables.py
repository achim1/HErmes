"""
Container classes for variables
"""

DEFAULT_BINS = 70
REGISTERED_FILEEXTENSIONS = [".h5",".root"]

import numpy as n
import inspect
import os
import pandas as pd
import tables
try:
    import root_numpy as rn
except ImportError:
    print "No root_numpy found, root support is limited!"
    REGISTERED_FILEEXTENSIONS.remove(".root")

from pyevsel.utils import files as f 
from pyevsel.utils.logger import Logger


################################################################

class Variable(object):
    """
    Container class holding variable
    properties
    """
    name  = None
    bins  = None
    label = None
    _is_harvested = False


    def __init__(self,name,bins=None,label="",transform=lambda x : x,definitions=[]):
        
        assert not (False in [len(x) <= 2 for x in definitions]), "Can not understand variable definitions %s!" %definitions
        if definitions:
            self.defsize = len(definitions[0])
            assert not (False in [len(x) == self.defsize for x in definitions]), "All definitions must have the same length!"
        else:
            self.defsize = 0

        self.name        = name
        self.bins        = bins # when histogrammed
        self.label       = label
        self.transform   = transform
        self.definitions = definitions
        if self.defsize  == 1:
            self.data    = pd.DataFrame()
        if self.defsize  == 2:
            self.data    = pd.Series()    
        self._is_harvested = False

    def __repr__(self):
        return """<Variable: %s>""" %self.name

    def calculate_fd_bins(self):
        """
        Calculate a reasonable binning
        """
        nbins = FreedmanDiaconisBins(self.data,min(self.data),max(self.data))
        self.bins = n.linspace(min(self.data),max(self.data),nbins)

    def harvest(self,*filenames):
        """
        Get the variable from a datafile
        """
        if self._is_harvested:
            return        

        for filename in filenames:
        
            ext = f.strip_all_endings(filename)[1]
            assert ext in REGISTERED_FILEEXTENSIONS, "Filetype %s not know" %ext
            assert os.path.exists(filename), "File %s does not exist!" %ext
            data = []
            defindex = 0
            if ext == ".h5":
                #store = pd.HDFStore(filename)
                store = tables.openFile(filename)

            found_data = False
            while not found_data:
                if defindex == len(self.definitions):
                    Logger.warning("No data for definitions %s found!" %self.definitions)
                    return
                if ext == ".h5":
                    if self.defsize == 2:
                        try:
                            #data = store.select_column(*self.definitions[defindex])
                            data = store.getNode("/" + self.definitions[defindex][0]).col(self.definitions[defindex][1])
                            data = pd.Series(data)
                            found_data = True
                        except AttributeError:
                            defindex += 1
                            continue
                    elif self.defsize == 1:
                        #data = store.select(self.definitions[defindex][0])
                        data = store.getNode("/" + self.definitions[defindex][0].read())
                        data = pd.DataFrame(data)
                elif ext == ".root":
                    #FIXME: What happens if it is not found in the rootfile
                    found_data = True #FIXME
                    data = rn.root2rec(filename,*self.definitions[defindex])
                    if self.defsize == 2:
                        data = pd.Series(data)
                    elif self.defsize == 1:
                        data = pd.DataFrame(data)
               
 
                self.data = self.data.append(data.map(self.transform))
                defindex += 1

            del data
        self._is_harvested = True
        return

##########################################################

class CompoundVariable(Variable):
    """
    A variable which can not be read out, but is calculated
    from other variables
    """

    def __init__(self,name,variables=[],label="",bins=None,operation=lambda x,y : x + y):
        self.name = name
        self.label = label
        self.bins = bins
        self._variables = variables
        self._operation = operation
        self.data = pd.Series()

    def _rewire_variables(self,vardict):
        """
        Use to avoid the necesity to read out variables twice
        as the variables are copied over by the categories, 
        the refernce is lost. Can be rewired though
        """
        newvars = []
        for var in self._variables:
            newvars.append(vardict[var.name])
        self._variables = newvars

    def __repr__(self):
        return """<CompoundVariable %s created from: %s>""" %(self.name,"".join([x.name for x in self._variables ]))

    def harvest(self):
        if self._is_harvested:
            return
        print self._variables
        harvestable = filter(lambda var : var._is_harvested, self._variables)
        if not len(harvestable) == len(self._variables):
            Logger.error("Variables have to be harvested for compound variable %s first!" %self.name)
            return
        self.data = reduce(self._operation,[var.data for var in self._variables])
        self._is_harvested = True

##########################################################

def FreedmanDiaconisBins(data,leftedge,rightedge,minbins=20,maxbins=70):
    """
    Get a number of bins for a histogram
    following Freedman/Diaconis
    """
    # default values
    bins = DEFAULT_BINS

    try:
        finite_data = n.isfinite(data)
        q3          = n.percentile(data[finite_data],75)
        q1          = n.percentile(data[finite_data],25)
        n_data      = len(data)
        h           = (2*(q3-q1))/(n_data**1./3)
        bins = (rightedge - leftedge)/h
    except Exception as e:
        Logger.warn("Calculate Freedman-Draconis bins failed %s" %e.__repr__())

    if (bins < minbins):
        bins = minbins
    elif  (bins > maxbins):
        bins = maxbins
    elif not n.isfinite(bins):
        Logger.warn("Calculate Freedman-Draconis bins failed, calculated nan bins, returning 70")
        bins = DEFAULT_BINS
    return bins

