"""
Container classes for variables
"""

DEFAULT_BINS = 70
REGISTERED_FILEEXTENSIONS = [".h5",".root"]

import numpy as n
import inspect
import os
import pandas as pd
import root_numpy as rn

from utils import files as f 



################################################################

class Variable(object):
    """
    Container class holding variable
    properties
    """
    def __init__(self,name,bins=None,label="",transform=lambda x : x,definitions=[]):
        
        assert not (False in [len(x) <= 2 for x in definitions]), "Can not understand variable definitions %s!" %definitions
        self.defsize = len(definitions[0])
        assert not (False in [len(x) == self.defsize for x in definitions]), "All definitions must have the same length!"

        self.name        = name
        self.bins        = bins # when histogrammed
        self.label       = label
        self.transform   = transform
        self.definitions = definitions
        if self.defsize  == 1:
            self.data    = pd.DataFrame()
        if self.defsize  == 2:
            self.data    = pd.Series()    

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
        
        for filename in filenames:
        
            ext = f.strip_all_endings(filename)[1]
            assert ext in REGISTERED_FILEEXTENSIONS, "Filetype %s not know" %ext
            assert os.path.exists(filename), "File %s does not exist!" %ext
            data = []
            defindex = 0
            while not len(data):
                if defindex == len(self.definitions):
                    raise ValueError("No data for definitions %s found!" %self.definitions)
                if ext == ".h5":
                    store = pd.HDFStore(filename)
                    if self.defsize == 2:
                        data = store.select_column(*self.definitions[defindex])
                    elif self.defsize == 1:
                        data = store.select(self.definitions[defindex][0])

                elif ext == ".root":
                    data = rn.root2rec(filename,*self.definitions[defindex])
                    if self.defsize == 2:
                        data = pd.Series(data)
                    elif self.defsize == 1:
                        data = pd.DataFrame(data)
    
                print ext,filename,data
                self.data = self.data.append(data.map(self.transform))
                defindex += 1

            del data



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

#####################################################

def GetVariablesFromModule(module):
    """
    Extract the variables from a python module with
    variable definitions in it
    """

    def cleaner(x):
        try:
            return not(x.__name__.startswith("_"))
        except:
            return False

    all_vars = inspect.getmembers(module)#,cleaner)
    print all_vars
    all_vars = [x[1] for x in all_vars if isinstance(x[1],Variable)]
    return all_vars

#####################################################


 

