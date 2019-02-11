"""
Container classes for variables
"""

from builtins import object
import numpy as n # remove that in the future
import numpy as np
import os
import pandas as pd
import tables
import abc
import enum

from ..utils import files as f
from ..utils.logger import Logger
from future.utils import with_metaclass

DEFAULT_BINS = 70

REGISTERED_FILEEXTENSIONS = [".h5"]

try:
    import uproot as ur
    REGISTERED_FILEEXTENSIONS.append(".root")

except ImportError:
    Logger.warning("No uproot found, root support is limited!")


################################################################
# define a non-member function so that it can be used in a
# multiprocessing approach
def extract_from_root(filename, definitions,
                      nevents=None,
                      reduce_dimension=None):
    """
    Use the uproot system to get information from rootfiles. Supports a basic tree of
    primitive datatype like structure.

    Args:
        filename (str): datafile
        defininitiions (list): tree and branch adresses

    Keyword Args:
        nevents (int): number of events to read out
        reduce_dimension (int): If data is vector-type, reduce it by taking the n-th element
    """

    can_be_concatted = False
    file = ur.open(filename)
    success = False
    i=0
    branch = None
    # it will most likely work only with TTrees
    while not success:
        try:
            tree = file.get(definitions[i][0])
            branch = tree
            for address in definitions[i][1:]:
                branch = branch.get(address)

            #tree = file[definitions[i][0]]
            #branch = file[definitions[i][0]].get(definitions[i][1])
            success = True
        except KeyError as e:
            Logger.warning("Can not find address {}".format(definitions[i]))
            i+=1
        except IndexError:
            Logger.critical("None of the provided keys could be found {}".format(definitions))
            break
    Logger.debug("Found valid definitions {}".format(definitions[i]))

    #FiXME make logger.critical end program!
    if nevents is not None:
        data = branch.array(entrystop=nevents)
    else:
        data = branch.array()

    # check for dimensionality
    multidim = False
    try:
        len(data[0])
        multidim = True
        Logger.debug("Assuming array data {}".format(definitions[i]))
    except TypeError:
        Logger.debug("Assuming scalar data {}".format(definitions[i]))

    if reduce_dimension is not None:
        if not multidim:
            raise ValueError("Can not reduce scalar data!")
        if isinstance(reduce_dimension, int):
            data = np.array([k[reduce_dimension] for k in data])
            multidim = False
        else:
            data = [[k[reduce_dimension[1]] for k in j] for j in data]

    if multidim:
        Logger.debug("Grabbing multidimensional data from root-tree for {}".format(definitions[i]))

        if nevents is None:
            data = branch.array() #this will be a jagged array now!
        else:
            data = branch.array(entrystop=nevents)

        tmpdata = [np.array(i) for i in data]
        # the below might not be picklable (multiprocessing!)
        #tmpdata = [i for i in data]
        # FIXME: dataframe/series
        # try to convert this to a pandas dataframe
        #data = pd.DataFrame(tmpdata)
        data = pd.Series(tmpdata)

        can_be_concatted = True
        #data.shape = (len(data),None)
        #data.ndim = 2
    else:
        Logger.debug("Grabbing scalar data from root-tree for {}".format(definitions[i]))
        data = pd.Series(np.asarray(data))
        can_be_concatted = True
    return data, can_be_concatted

################################################################
# define a non-member function so that it can be used in a
# multiprocessing approach
def harvest(filenames, definitions, **kwargs):
    """
    Read variables from files into memory. Will be used by HErmes.selection.variables.Variable.harvest
    This will be run multi-threaded. Keep that in mind, arguments have to be picklable,
    also everything thing which is read out must be picklable. Lambda functions are NOT picklable

    Args:
        filenames (list): the files to extract the variables from.
                          currently supported: hdf
        definitions (list): where to find the data in the files. They usually
                            have some tree-like structure, so this a list
                            of leaf-value pairs. If there is more than one
                            all of them will be tried. (As it might be that
                            in some files a different naming scheme was used)
                            Example: [("hello_reoncstruction", "x"), ("hello_reoncstruction", "y")] ]
    Keyword Args:
        transformation (func): After the data is read out from the files,
                               transformation will be applied, e.g. the log
                               to the energy.
        fill_empty (bool): Fill empty fields with zeros
        nevents (int): ROOT only - read out only nevents from the files
        reduce_dimension (str): ROOT only - multidimensional data can be reduced by only
                                            using the index given by reduce_dimension.
                                            E.g. in case of a TVector3, and we want to have onlz
                                            x, that would be 0, y -> 1 and z -> 2.

        FIXME: Not implemented yet! precision (int): Precision in bit

    Returns:
        pd.Series or pd.DataFrame
    """

    nevents = kwargs["nevents"] if "nevents" in kwargs else None
    fill_empty = kwargs["fill_empty"] if "fill_empty" in kwargs else False
    reduce_dimension = kwargs["reduce_dimension"] if "reduce_dimension" in kwargs else None
    transform = kwargs["transformation"] if "transformation" in kwargs else None
    concattable = True
    data = pd.Series()
    #multidim_data = pd.DataFrame()
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

            definition = list(definition) 
            if filetype == ".h5":
                if not definition[0].startswith("/"):
                    definition[0] = "/" + definition[0]
                try:
                    # data = store.select_column(*definition)
                    if not definition[1]:
                        tmpdata = hdftable.get_node(definition[0])
                    else:
                        tmpdata = hdftable.get_node(definition[0]).col(definition[1])
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
                tmpdata, concattable = extract_from_root(filename, definitions,
                                                         nevents=nevents,
                                                         reduce_dimension=reduce_dimension)
        if filetype == ".h5":
            hdftable.close()

        #tmpdata = harvest_single_file(filename, filetype,definitions)
        # self.data = self.data.append(data.map(self.transform))
        # concat should be much faster
        if not concattable:
            logger.warn("Data can not be concatted, keep that in mind!")
            return tmpdata

        data = pd.concat([data, tmpdata])
        del tmpdata
    #print (data)
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


        if not n.isfinite(bins):
            Logger.warn("Calculate Freedman-Draconis bins failed, calculated nan bins, returning fallback")
            bins = fallbackbins

        if bins < minbins:
            bins = minbins
        if bins > maxbins:
            bins = maxbins
    except Exception as e:
        Logger.warn("Calculate Freedman-Draconis bins failed {0}".format( e.__repr__()))
        bins = fallbackbins
    return bins

#############################################################

class VariableRole(enum.Enum):
    """
    Define roles for variables. Some variables used in a special context (like weights)
    are easily recognizable by this flag.
    """
    UNKNOWN         = 0
    SCALAR          = 10
    ARRAY           = 20
    GENERATORWEIGHT = 30
    RUNID           = 40
    EVENTID         = 50
    STARTIME        = 60
    ENDTIME         = 70



##############################################################

class AbstractBaseVariable(with_metaclass(abc.ABCMeta, object)):
    """
    Read out tagged numerical data from files
    """    
    _harvested = False
    _bins = None
    ROLES = VariableRole

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
        self._harvested = True

    @property
    def harvested(self):
        return self._harvested

    @property
    def bins(self):
        if self._bins is None:
            return self.calculate_fd_bins()
        else:
            return self._bins

    @bins.setter
    def bins(self, value):
        self._bins = value

    def calculate_fd_bins(self):
        """
        Calculate a reasonable binning

        Returns:
            numpy.ndarray: Freedman Diaconis bins
        """
        nbins = freedman_diaconis_bins(self.data, min(self.data), max(self.data))
        self.bins = n.linspace(min(self.data),max(self.data), nbins)
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
    def ndim(self):
        return self._data.ndim

    @property
    def data(self):
        if isinstance(self._data, pd.DataFrame):
            return self._data.as_matrix()
        if not hasattr(self._data, "shape"):
            Logger.warning("Something's wrong, this should be array data!")
            Logger.warning("Seeeing {} data".format(type(self._data)))
            Logger.warning("Attempting to fix!")
            self._data = np.asarray(self._data)
            return self._data

        return self._data

############################################

class Variable(AbstractBaseVariable):
    """
    A hook to a single variable read out from a file
    """

    def __init__(self, name, definitions=None,\
                 bins=None, label="", transform=lambda x: x,
                 role=VariableRole.SCALAR,
                 nevents=None,
                 reduce_dimension=None):
        """
        Args:
            name (str): An unique identifier

        Keyword Args:
            definitions (list): table and/or column names in underlying data
            bins (numpy.ndarray): used for histograms
            label (str): used for plotting and as a label in tables
            transform (func): apply to each member of the underlying data at readout
            role (HErmes.selection.variables.VariableRole): The role the variable is playing. In most cases the default is the best choice
            nevents (int): number of events to read in (ROOT only right now!)
            reduce_dimension (int): in case of multidimensionality, take only the the given index of the array (ROOT only right now)
        """
        AbstractBaseVariable.__init__(self)

        if definitions is not None:
            #assert not (False in [len(x) <= 2 for x in definitions]), "Can not understand variable definitions {}!".format(definitions)
            self.defsize = len(definitions[0])
            assert not (False in [len(x) == self.defsize for x in definitions]), "All definitions must have the same length!"
        else:
            self.defsize = 0

        self.name        = name
        self.role        = role
        self.bins        = bins # when histogrammed
        self.label       = label
        self.transform   = transform
        self.definitions = definitions
        self._data       = pd.Series()
        self.nevents     = nevents
        self.reduce_dimension = reduce_dimension

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
    Calculate a variable from other variables. This kind of variable will not read any file.
    """

    def __init__(self, name, variables=None, label="",\
                 bins=None, operation=lambda x,y : x + y,
                 role=VariableRole.SCALAR):
        """
        Args:
            name (str): An unique identifier for the new variable.

        Keyword Args:
            variables (list): A list of variables used to calculate the new variable.
            label (str): A label for plotting.
            bins (np.ndarray): binning for distributions.
            operation (fnc): The operation which will be applied to variables.
            role (HErmes.selection.variables.VariableRole): The role the variable is playing. In most cases the default is the best choice
        """
        AbstractBaseVariable.__init__(self)
        self.name = name
        self.role = role
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

    def harvest(self, *filenames):
        #FIXME: filenames is not used, just
        #there for compatibility

        if self.harvested:
            return
        harvested = [var for var in self.variables if var.harvested]
        if not len(harvested) == len(self.variables):
            Logger.error("Variables have to be harvested for compound variable {0} first!".format(self.variables))
            Logger.error("Only {} is harvested".format(harvested))
            return
        self._data = self.operation(*[var.data for var in self.variables])
        self.declare_harvested()

##########################################################


class VariableList(AbstractBaseVariable):
    """
    A list of variable. Can not be read out from files.
    """

    def __init__(self, name, variables=None, label="", bins=None, role=VariableRole.SCALAR):
        """
        Args:
            name (str): An unique identifier for the new category.

        Keyword Args:
            variables (list): A list of variables used to calculate the new variable.
            label (str): A label for plotting.
            bins (np.ndarray): binning for distributions.
            role (HErmes.selection.variables.VariableRole): The role the variable is playing. In most cases the default is the best choice
        """

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

        # do not calculate weights yet!


        if self.harvested:
            return
        harvested = [var for var in self.variables if var.harvested]
        if not len(harvested) == len(self.variables):
            Logger.error("Variables have to be harvested for compound variable {} first!".format(self.name))
            return
        self.declare_harvested()

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

    @property
    def data(self):
        return [x.data for x in self.variables]



