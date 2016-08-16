"""
Container classes for variables
"""

import numpy as n
import os
import pandas as pd
import tables
import abc

from pyevsel.utils import files as f
from pyevsel.utils.logger import Logger


DEFAULT_BINS = 70
REGISTERED_FILEEXTENSIONS = [".h5",".root"]

try:
    import root_numpy as rn
except ImportError:
    Logger.warning("No root_numpy found, root support is limited!")
    REGISTERED_FILEEXTENSIONS.remove(".root")

################################################################
# define some non-member function so that they can be used in a
# multiprocessing approach

def harvest_single_file(filename, filetype, definitions):
    """
    Get the variable data from a fileobject
    Optimized for hdf files

    Args:
        filename (str):
        filetype (str): the extension of the filename, eg "h5"

    Returns:
        pd.Series
    """
    if filetype == ".h5" and not isinstance(filename, tables.table.Table):
        # store = pd.HDFStore(filename)
        hdftable = tables.openFile(filename)

    else:
        hdftable = filename

    data = pd.Series()
    for definition in definitions:
        if filetype == ".h5":
            try:
                # data = store.select_column(*definition)
                data = hdftable.getNode("/" + definition[0]).col(definition[1])
                data = pd.Series(data, dtype=n.float64)
                break
            except tables.NoSuchNodeError:
                Logger.debug("Can not find definition {0} in {1}! ".format(definition, filename))
                continue

        elif filetype == ".root":
            data = rn.root2rec(filename, *definition)
            data = pd.Series(data)
    if filetype == ".h5":
        hdftable.close()

    return data


def harvest(filenames,definitions,**kwargs):
    """
    Extract the variable data from the provided files

    Args:
        filenames (list): the files to extract from
                          currently supported: {0}

    Keyword Args:
        transformation (func): will be applied to the read out data

    Returns:
        pd.Series or pd.DataFrame
    """.format(REGISTERED_FILEEXTENSIONS.__repr__())

    data = pd.Series()
    for filename in filenames:
        ext = f.strip_all_endings(filename)[1]
        assert ext in REGISTERED_FILEEXTENSIONS, "Filetype {} not known!".format(ext)
        assert os.path.exists(filename), "File {} does not exist!".format(ext)
        Logger.debug("Attempting to harvest {1} file {0}".format(filename,ext))
        
        tmpdata = harvest_single_file(filename, ext,definitions)
        # self.data = self.data.append(data.map(self.transform))
        # concat should be much faster
        if "transformation" in kwargs:
            transform = kwargs['transformation']
            data = pd.concat([data, tmpdata.map(transform)])
        else:
            data = pd.concat([data, tmpdata])
        del tmpdata
    return data

################################################################

class AbstractBaseVariable(object):
    """
    A 'variable' is a large set of numerical data
    stored in some file or database.
    This class purpose is to read this data
    and load it into memory so that it cna be
    used with pandas/numpy
    """    


    __metaclass__ = abc.ABCMeta
    _is_harvested = False

    def __hash__(self):
        return self.name

    def __repr__(self):
        return """<Variable: {}>""".format(self.name)

    def __eq__(self,other):
        return self.name == other.name

    def __lt__(self, other):
        return sorted(self.name,other.name)[0] == self.name

    def __gt__(self, other):
        return sorted(self.name,other.name)[1] == self.name

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    def declare_harvested(self):
        self._is_harvested = True

    def undeclare_harvested(self):
        self._is_harvested = False

    @property
    def is_harvested(self):
        return self._is_harvested

    def calculate_fd_bins(self):
        """
        Calculate a reasonable binning

        Returns:
            numpy.ndarray: Freedman Diaconis bins
        """
        nbins = freedman_diaconis_bins(self.data,min(self.data),max(self.data))
        self.bins = n.linspace(min(self.data),max(self.data),nbins)
        return self.bins

    @abc.abstractmethod
    def harvest(self,*files):
        """
        Read the data from the provided files

        Args:
            *files: walk through these files and readout

        Returns:
            None
        """
        return

class Variable(AbstractBaseVariable):
    """
    Container class holding variable
    properties
    """

    def __init__(self,name,definitions=None,bins=None,label="",transform=lambda x : x):
        """
        Create a new variable

        Args:
            name (str): An unique identifier

        Keyword Args:
            definitions (list): table and/or column names in underlying data
            bins (numpy.ndarray): used for histograms
            label (str): used for plotting and as a label in tables
            transform (func): apply to each member of the underlying data at readout
        """
        AbstractBaseVariable.__init__(self)

        if definitions is not None:
            assert not (False in [len(x) <= 2 for x in definitions]), "Can not understand variable definitions {}!".format(definitions)
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

    def harvest_from_rootfile(self,rootfile,definition):
        """
        Get data from a root file

        Args:
            rootfile (str): Name of the *.root file
            definition (tuple): Name of branches/leaves in the rootfile
        Returns:
            pd.Series or DataFrame
        """
        #FIXME: What happens if it is not found in the rootfile

        data = rn.root2rec(rootfile,*definition)
        if self.defsize == 2:
            data = pd.Series(data)
        elif self.defsize == 1:
            data = pd.DataFrame(data)
        else:
            raise ValueError
        return data

    def harvest_from_hdftable(self,hdftable,definition):
        """
        Get the data from a pre-opened hdf file

        Args:
            hdftable (tables.Table): open hdf file
            definition (tuple): names of tables in hdftable

        Returns:
            pd.Series or pd.DataFrame
        """

        if self.defsize == 2:
            try:
                #data = store.select_column(*definition)
                data = hdftable.getNode("/" + definition[0]).col(definition[1])
                data = pd.Series(data,dtype=n.float64)
            except tables.NoSuchNodeError:
                return None
        elif self.defsize == 1: #FIXME what happens if it isn't found?
            #data = store.select(self.definitions[defindex][0])
            data = hdftable.getNode("/" + definition[0].read())
            data = pd.DataFrame(data)
        return data

    def harvest_single_file(self,fileobject,filetype):
        """
        Get the variable data from a fileobject
        Optimized for hdf files

        Args:
            fileobject (str or hdfNode):
            filetype (str): the extension of the filename, eg "h5"

        Returns:
            pd.Series or pd.Dataframe
        """
        if filetype == ".h5" and not isinstance(fileobject,tables.table.Table):
            #store = pd.HDFStore(filename)
            store = tables.openFile(fileobject)

        for definition in self.definitions:
            if filetype == ".h5":
                data = self.harvest_from_hdftable(store,definition)
                if data is None:
                    #Logger.debug("Can not find definition {0} in {1}! ".format(definition,fileobject))
                    continue
                else:
                    break

            elif filetype == ".root":
                data = self.harvest_from_rootfile(fileobject,definition)

        if filetype == ".h5":
            store.close()

        #FIXME: rework this and return always
        # the same stuff
        return data

    def harvest(self,*filenames):
        """
        Extract the variable data from the provided files

        Args:
            filenames (list): the files to extract from
                              currently supported: {}

        Returns:
            pd.Series or pd.DataFrame
        """.format(REGISTERED_FILEEXTENSIONS.__repr__())
        if self.is_harvested:
            return

        data = pd.Series()
        if self.defsize == 1:
            data = pd.DataFrame()

        for filename in filenames:
            ext = f.strip_all_endings(filename)[1]
            assert ext in REGISTERED_FILEEXTENSIONS, "Filetype {} not known!".format(ext)
            assert os.path.exists(filename), "File {} does not exist!".format(filename)
            #Logger.debug("Attempting to harvest file {0}".format(filename))
            data = self.harvest_single_file(filename,ext)
            #self.data = self.data.append(data.map(self.transform))
            #concat should be much faster

            if isinstance(data,pd.Series):
                self.data = pd.concat([self.data,data.map(self.transform)])
            else:
                self.data = pd.concat([self.data,data])
            del data

        self.declare_harvested()
        return None

##########################################################

class CompoundVariable(AbstractBaseVariable):
    """
    A variable which can not be read out, but is calculated
    from other variables
    """

    def __init__(self,name,variables=None,label="",\
                 bins=None,operation=lambda x,y : x + y):
        AbstractBaseVariable.__init__(self)
        self.name = name
        self.label = label
        self.bins = bins
        if variables is None:
            variables = []
        self.variables = variables
        self.operation = operation
        self.data = pd.Series()

    def rewire_variables(self,vardict):
        """
        Use to avoid the necessity to read out variables twice
        as the variables are copied over by the categories, 
        the refernce is lost. Can be rewired though
        """
        newvars = []
        for var in self.variables:
            newvars.append(vardict[var.name])
        self.variables = newvars

    def __repr__(self):
        return """<CompoundVariable {} created from: {}>""".format(self.name,"".join([x.name for x in self._variables ]))

    def harvest(self,*filenames):
        #FIXME: filenames is not used, just
        #there for compatibility

        if self.is_harvested:
            return
        harvested = filter(lambda var : var.is_harvested, self.variables)
        if not len(harvested) == len(self.variables):
            Logger.error("Variables have to be harvested for compound variable {0} first!".format(self.name))
            return
        self.data = reduce(self.operation,[var.data for var in self.variables])
        self.declare_harvested()

##########################################################


class VariableList(AbstractBaseVariable):
    """
    Holds several variable values
    """

    def __init__(self,name,variables=None,label="",bins=None):
        AbstractBaseVariable.__init__(self)
        self.name = name
        self.label = label
        self.bins = bins
        if variables is None:
            variables = []
        self.variables = variables

    def harvest(self,*filenames):
        #FIXME: filenames is not used, just
        #there for compatibility

        if self.is_harvested:
            return
        harvested = filter(lambda var : var.is_harvested, self.variables)
        if not len(harvested) == len(self.variables):
            Logger.error("Variables have to be harvested for compound variable {} first!".format(self.name))
            return
        self.declare_harvested()

    def rewire_variables(self,vardict):
        """
        Use to avoid the necesity to read out variables twice
        as the variables are copied over by the categories,
        the refernce is lost. Can be rewired though
        """
        newvars = []
        for var in self.variables:
            newvars.append(vardict[var.name])
        self.   variables = newvars

    @property
    def data(self):
        return [x.data for x in self.variables]

##########################################################


def freedman_diaconis_bins(data,leftedge,\
                         rightedge,minbins=20,\
                         maxbins=70,fallbackbins=DEFAULT_BINS):
    """
    Get a number of bins for a histogram
    following Freedman/Diaconis

    Args:
        leftedge (float): left bin edge
        rightedge (float): right bin edge
        minbins (int): the minimum number of bins
        maxbins (int): the maximum number of bins
        fallbackbins (int): a number of bins which is returned
                            if calculation failse

    Returns:
        nbins (int): number of bins, minbins < bins < maxbins
    """

    try:
        finite_data = n.isfinite(data)
        q3          = n.percentile(data[finite_data],75)
        q1          = n.percentile(data[finite_data],25)
        n_data      = len(data)
        h           = (2*(q3-q1))/(n_data**1./3)
        bins = (rightedge - leftedge)/h
    except Exception as e:
        Logger.warn("Calculate Freedman-Draconis bins failed {0}".format( e.__repr__()))
        bins = fallbackbins

    if not n.isfinite(bins):
        Logger.warn("Calculate Freedman-Draconis bins failed, calculated nan bins, returning fallback")
        bins = fallbackbins

    if bins < minbins:
        bins = minbins
    if bins > maxbins:
        bins = maxbins

    return bins

