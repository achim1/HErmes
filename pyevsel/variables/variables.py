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
from pyevsel.utils import GetTiming

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

        FIXME: this entire method needs to be reworked!
        """
        if self._is_harvested:
            return        

        for filename in filenames:
            ext = f.strip_all_endings(filename)[1]
            assert ext in REGISTERED_FILEEXTENSIONS, "Filetype %s not know" %ext
            assert os.path.exists(filename), "File %s does not exist!" %ext
            data = pd.Series()
            if self.defsize == 1:
                data = pd.DataFrame()
            #defindex = 0
            if ext == ".h5":
                #store = pd.HDFStore(filename)
                store = tables.openFile(filename)

            for definition in self.definitions:
                #if defindex == len(self.definitions):
                #    Logger.warning("No data for definitions %s found!" %self.definitions)
                #    return
                if ext == ".h5":
                    if self.defsize == 2:
                        try:
                            #data = store.select_column(*definition)
                            data = store.getNode("/" + definition[0]).col(definition[1])
                            data = pd.Series(data)
                            #found_data = True
                        except AttributeError:
                            #defindex += 1
                            continue
                    elif self.defsize == 1: #FIXME what happens if it isn't found?
                        #data = store.select(self.definitions[defindex][0])
                        data = store.getNode("/" + definition[0].read())
                        data = pd.DataFrame(data)
                        #found_data = True
                elif ext == ".root":
                    #FIXME: What happens if it is not found in the rootfile
                    #found_data = True #FIXME
                    data = rn.root2rec(filename,*definition)
                    if self.defsize == 2:
                        data = pd.Series(data)
                    elif self.defsize == 1:
                        data = pd.DataFrame(data)

                #defindex += 1

            if ext == ".h5":
                store.close()

            #self.data = self.data.append(data.map(self.transform))
            #concat should be much faster
            if not len(self.data):
                if isinstance(data,pd.Series):
                    self.data = data.map(self.transform)
                else:
                    self.data = data

            else:
                if isinstance(data,pd.Series):
                    self.data = pd.concat([self.data,data.map(self.transform)])
                else:
                    self.data = pd.concat([self.data,data])
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

    def harvest(self,*filenames):
        #FIXME: filenames is not used, just
        #there for compatibility

        if self._is_harvested:
            return
        harvested = filter(lambda var : var._is_harvested, self._variables)
        if not len(harvested) == len(self._variables):
            Logger.error("Variables have to be harvested for compound variable %s first!" %self.name)
            return
        self.data = reduce(self._operation,[var.data for var in self._variables])
        self._is_harvested = True

##########################################################

class VariableList(Variable):
    """
    Holds several variable values
    """

    def __init__(self,name,variables=[],label="",bins=None,operation=lambda x,y : x + y):
        self.name = name
        self.label = label
        self.bins = bins
        self._variables = variables


    def harvest(self,*filenames):
        #FIXME: filenames is not used, just
        #there for compatibility

        if self._is_harvested:
            return
        harvested = filter(lambda var : var._is_harvested, self._variables)
        if not len(harvested) == len(self._variables):
            Logger.error("Variables have to be harvested for compound variable %s first!" %self.name)
            return
        self._is_harvested = True


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

    @property
    def data(self):
        return [x.data for x in self._variables]

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

