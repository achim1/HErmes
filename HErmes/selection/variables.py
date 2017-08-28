"""
Container classes for variables
"""

from builtins import object
import numpy as n
import os
import pandas as pd
import tables
import abc

from ..utils import files as f
from ..utils.logger import Logger
from future.utils import with_metaclass
from functools import reduce


DEFAULT_BINS = 70

# FIXME: add root support!
# REGISTERED_FILEEXTENSIONS = [".h5", ".root"]
REGISTERED_FILEEXTENSIONS = [".h5"]

try:
    import root_numpy as rn
except ImportError:
    Logger.warning("No root_numpy found, root support is limited!")
    if ".root" in REGISTERED_FILEEXTENSIONS:
        REGISTERED_FILEEXTENSIONS.remove(".root")

################################################################
# define a non-member function so that it can be used in a
# multiprocessing approach

def harvest(filenames, definitions, **kwargs):
    """
    Extract the variable data from the provided files

    Args:
        filenames (list): the files to extract the variables from.
                          currently supported: {0}
        definitions (list): where to find the data in the files. They usually
                            have some tree-like structure, so this a list
                            of leaf-value pairs. If there is more than one
                            all of them will be tried. (As it might be that
                            in some files a different naming scheme was used)
                            Example: [("hello_reoncstruction", "x"), ("helo_reoncstruction", "x")] ]
    Keyword Args:
        transformation (func): After the data is read out from the files,
                               transformation will be applied, e.g. the log
                               to the energy.
        fill_empty (bool): Fill empty fields with zeros


        FIXME: Not implemented yet! precision (int): Precision in bit

    Returns:
        pd.Series or pd.DataFrame
    """.format(REGISTERED_FILEEXTENSIONS.__repr__())

    fill_empty = kwargs["fill_empty"] if "fill_empty" in kwargs else False
    data = pd.Series()
    for filename in filenames:
        filetype = f.strip_all_endings(filename)[1]

        assert filetype in REGISTERED_FILEEXTENSIONS, "Filetype {} not known!".format(filetype)
        assert os.path.exists(filename), "File {} does not exist!".format(filetype)
        Logger.debug("Attempting to harvest {1} file {0}".format(filename,filetype))
        
        if filetype == ".h5" and not isinstance(filename, tables.table.Table):
            # store = pd.HDFStore(filename)
            hdftable = tables.open_file(filename)

        else:
            hdftable = filename

        tmpdata = pd.Series()
        for definition in definitions:
            if filetype == ".h5":
                try:
                    # data = store.select_column(*definition)
                    tmpdata = hdftable.get_node("/" + definition[0]).col(definition[1])
                    if tmpdata.ndim == 2:
                        if data.empty:
                            data = pd.DataFrame()
                        tmpdata = pd.DataFrame(tmpdata, dtype=n.float32)
                    else:
                        tmpdata = pd.Series(tmpdata, dtype=n.float32)

                    Logger.debug("Found {} entries in table for {}{}".format(len(tmpdata),definition[0],definition[1]))
                    break
                except tables.NoSuchNodeError:
                    Logger.debug("Can not find definition {0} in {1}! ".format(definition, filename))
                    continue

            elif filetype == ".root":
                tmpdata = rn.root2rec(filename, *definition)
                tmpdata = pd.Series(data)
        if filetype == ".h5":
            hdftable.close()

        #tmpdata = harvest_single_file(filename, filetype,definitions)
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

#############################################################

class AbstractBaseVariable(with_metaclass(abc.ABCMeta, object)):
    """
    A 'variable' is a large set of numerical data
    stored in some file or database.
    This class purpose is to read this data
    and load it into memory so that it cna be
    used with pandas/numpy
    """    
    _is_harvested = False

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return """<Variable: {}>""".format(self.name)

    def __eq__(self,other):
        return self.name == other.name

    def __lt__(self, other):
        return sorted([self.name,other.name])[0] == self.name

    def __gt__(self, other):
        return sorted([self.name,other.name])[1] == self.name

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

    def harvest(self, *files):
        """
        Hook to the harvest method. Don't use in case of multiprocessing!
        Args:
            *files: walk through these files and readout
        """

        self._data = harvest(files, self.definitions)
        self.declare_harvested()

    @abc.abstractmethod
    def rewire_variables(self, vardict):
        return

    @property
    def data(self):
        if isinstance(self._data, pd.DataFrame):
            return self._data.as_matrix()
        else:
            return self._data

############################################

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
        self._data       = pd.Series()
        #if self.defsize  == 1:
        #    self.data    = pd.DataFrame()
        #if self.defsize  == 2:
        #    self.data    = pd.Series()

    def rewire_variables(self, vardict):
        """
        Make sure all the variables are connected properly. This is
        only needed for combined/compound variables

        Returns:
            None
        """

        pass

##########################################################

class CompoundVariable(AbstractBaseVariable):
    """
    Has no representation in any file, but is calculated
    by other variables
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
        self._data = pd.Series()
        self.definitions = ((self.__repr__()),)

    def rewire_variables(self, vardict):
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
        return """<CompoundVariable {} created from: {}>""".format(self.name,"".join([x.name for x in self.variables ]))

    def harvest(self,*filenames):
        #FIXME: filenames is not used, just
        #there for compatibility

        if self.is_harvested:
            return
        harvested = [var for var in self.variables if var.is_harvested]
        if not len(harvested) == len(self.variables):
            Logger.error("Variables have to be harvested for compound variable {0} first!".format(self.variables))
            Logger.error("Only {} is harvested".format(harvested))
            return
        #self.data = reduce(self.operation,[var.data for var in self.variables])
        self._data = self.operation(*[var.data for var in self.variables])
        self.declare_harvested()

##########################################################


class VariableList(AbstractBaseVariable):
    """
    Holds several variable values
    """

    def __init__(self, name, variables=None, label="", bins=None):
        AbstractBaseVariable.__init__(self)
        self.name = name
        self.label = label
        self.bins = bins
        if variables is None:
            variables = []
        self.variables = variables

    def harvest(self, *filenames):
        #FIXME: filenames is not used, just
        #there for compatibility

        if self.is_harvested:
            return
        harvested = [var for var in self.variables if var.is_harvested]
        if not len(harvested) == len(self.variables):
            Logger.error("Variables have to be harvested for compound variable {} first!".format(self.name))
            return
        self.declare_harvested()

    def rewire_variables(self, vardict):
        """
        Use to avoid the necesity to read out variables twice
        as the variables are copied over by the categories,
        the refernce is lost. Can be rewired though
        """
        newvars = []
        for var in self.variables:
            newvars.append(vardict[var.name])
        self.variables = newvars

    @property
    def data(self):
        return [x.data for x in self.variables]



