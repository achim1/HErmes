"""
Container classes for variables
"""

import numpy as np
import os
import pandas as pd
import tables
import abc
import enum
import array
import numbers
from ..utils import files as f
from ..utils import Logger
from copy import deepcopy as copy
DEFAULT_BINS = 70

REGISTERED_FILEEXTENSIONS = [".h5"]

try:
    import uproot as ur
    import uproot_methods.classes.TVector3 as TVector3
    import uproot_methods.classes.TLorentzVector as TLorentzVector
    import uproot_methods.classes.TH1
    REGISTERED_FILEEXTENSIONS.append(".root")

except ImportError:
    Logger.warning("No uproot found, root support is limited!")


################################################################
# define a non-member function so that it can be used in a
# multiprocessing approach
def extract_from_root(filename, definitions,
                      nevents=None,
                      dtype=np.float64,
                      transform = None,
                      reduce_dimension=None):
    """
    Use the uproot system to get information from rootfiles. Supports a basic tree of
    primitive datatype like structure.

    Args:
        filename (str): datafile
        definitiions (list): tree and branch adresses

    Keyword Args:
        nevents (int): number of events to read out
        reduce_dimension (int): If data is vector-type, reduce it by taking the n-th element
        dtype (np.dtyoe):  A numpy datatype, default double (np.float64) - use smaller dtypes to 
                           save memory
        transform (func): A function which directy transforms the readout data
    """

    can_be_concatted = False
    rootfile = ur.open(filename)
    success = False
    i=0
    branch = None
    # it will most likely work only with TTrees
    while not success:
        try:
            tree = rootfile.get(definitions[i][0])
            branch = tree
            for address in definitions[i][1:]:
                Logger.debug("Searching for address {}".format(address))
                branch = branch.get(address)

            #tree = file[definitions[i][0]]
            #branch = rootfile[definitions[i][0]].get(definitions[i][1])
            success = True
        except KeyError as e:
            Logger.warning(f"Can not find address {definitions[i]}")
            i+=1
        except IndexError:
            Logger.critical(f"None of the provided keys could be found {definitions}")
            break
    Logger.debug(f"Found valid definitions {definitions[i]}")

    ##FiXME make logger.critical end program!
    if nevents is not None:
        data = branch.array(entrystop=nevents)
    else:
        data = branch.array()

    # check for dimensionality
    multidim = False
    try:
        len(data[0])
        multidim = True
    except TypeError:
        Logger.debug(f"Assuming scalar data {definitions[i]}")

    if multidim:
        Logger.debug("Inspecting data...")
        tmp_lengths = set([len(k) for k in data])
        Logger.debug("Found {}".format(tmp_lengths))
        firstlen = list(tmp_lengths)[0]
        if (len(tmp_lengths) == 1) and (firstlen == 1):
            multidim = False
            Logger.debug("Found data containing iterables of size 1... flattening!")
            del tmp_lengths
            if dtype != np.float64:
                tmpdata = array.array("f",[])
            else:
                tmpdata = array.array("d",[])
            if isinstance(data[0][0], numbers.Number):
                [tmpdata.append(dtype(k)) for k in data]
                #tmpdata = np.asarray([k[0] for k in data])
                #multidim = True
                data = tmpdata
                del tmpdata 
            else:
                Logger.info("Is multidim data")
                multidim = True
        else:
            del tmp_lengths
            multidim = True
            Logger.debug("Assuming array data {}".format(definitions[i]))

    if reduce_dimension is not None:
        if not multidim:
            raise ValueError("Can not reduce scalar data!")
        if isinstance(reduce_dimension, int):
            data = np.array([k[reduce_dimension] for k in data], dtype=dtype)
            multidim = False
        else:
            data = [[k[reduce_dimension[1]] for k in j] for j in data]
    if multidim:
        Logger.debug("Grabbing multidimensional data from root-tree for {}".format(definitions[i]))
        del data
        if nevents is None:
            data = branch.array() #this will be a jagged array now!
        else:
            data = branch.array(entrystop=nevents)
        del branch
        if (len(data[0])):
            if isinstance(data[0][0], TVector3.TVector3):
                Logger.info("Found TVector3 data, treating appropriatly")
                data =  pd.Series([np.array([i.x,i.y,i.z], dtype=dtype) for i in data])
            if isinstance(data[0][0], TLorentzVector.TLorentzVector):
                Logger.info("Found TLorentzVector data, treating appropriatly")
                data =  pd.Series([np.array([i.x,i.y,i.z,i.t], dtype=dtype) for i in data])
        else: # probably number then    
            data = pd.Series([np.asarray(i,dtype=dtype) for i in data])
    
        # the below might not be picklable (multiprocessing!)
        #tmpdata = [i for i in data]
        # FIXME: dataframe/series
        # try to convert this to a pandas dataframe
        #data = pd.DataFrame(tmpdata)
        can_be_concatted = True
        data.multidim = True
    else:
        Logger.debug("Grabbing scalar data from root-tree for {}".format(definitions[i]))
        # convert in cases of TVector3/TLorentzVector

        if isinstance(data[0], TVector3.TVector3):
            Logger.debug("Found TVector3")
            data =  pd.Series([np.array([i.x,i.y,i.z], dtype=dtype) for i in data])
        elif isinstance(data[0], TLorentzVector.TLorentzVector):
            Logger.debug("Found TLorentzVector")
            data =  pd.Series([np.array([i.x,i.y,i.z, i.t], dtype=dtype) for i in data])
        else:
            try:
                #FIXME: why is that asarray needed?
                #data = pd.Series(np.asarray(data,dtype=dtype))
                data = pd.Series(data,dtype=dtype)
            except TypeError: # data consist of some object
                data = pd.Series(data) 
        Logger.debug("Got {} elements for {}".format(len(data), definitions[i]))
        can_be_concatted = True
    if transform is not None:
        data = transform(data)
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
        dtype (np.dtype) : datatype to cast to (default np.float64, but can be used
                           to reduce memory footprint.

    Returns:
        pd.Series or pd.DataFrame
    """

    nevents          = kwargs["nevents"] if "nevents" in kwargs else None
    fill_empty       = kwargs["fill_empty"] if "fill_empty" in kwargs else False
    reduce_dimension = kwargs["reduce_dimension"] if "reduce_dimension" in kwargs else None
    transform        = kwargs["transformation"] if "transformation" in kwargs else None
    dtype            = kwargs["dtype"] if "dtype" in kwargs else np.float64

    concattable = True
    data = pd.Series(dtype=dtype)
    #multidim_data = pd.DataFrame()
    for filename in filenames:
        filetype = f.strip_all_endings(filename)[1]

        assert filetype in REGISTERED_FILEEXTENSIONS, "Filetype {} not known!".format(filetype)
        assert os.path.exists(filename), "File {} does not exist!".format(filetype)
        if (filetype == ".h5") and (transform is not None):
            Logger.critical("Can not apply direct transformation for h5 files (yet). This is only important for root files and varaibles which are used as VariableRole.PARAMETER")
        Logger.debug("Attempting to harvest {1} file {0}".format(filename,filetype))
        
        if filetype == ".h5" and not isinstance(filename, tables.table.Table):
            # store = pd.HDFStore(filename)
            hdftable = tables.open_file(filename)

        else:
            hdftable = filename

        tmpdata = pd.Series(dtype=dtype)
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
                        tmpdata = pd.DataFrame(tmpdata, dtype=dtype)
                    else:
                        tmpdata = pd.Series(tmpdata, dtype=dtype)

                    Logger.debug("Found {} entries in table for {}{}".format(len(tmpdata),definition[0],definition[1]))
                    break
                except tables.NoSuchNodeError:
                    Logger.debug("Can not find definition {0} in {1}! ".format(definition, filename))
                    continue

            elif filetype == ".root":
                tmpdata, concattable = extract_from_root(filename, definitions,
                                                         nevents=nevents,
                                                         dtype=dtype,
                                                         transform=transform,
                                                         reduce_dimension=reduce_dimension)
        if filetype == ".h5":
            hdftable.close()

        #tmpdata = harvest_single_file(filename, filetype,definitions)
        # self.data = self.data.append(data.map(self.transform))
        # concat should be much faster
        if not True in [isinstance(tmpdata, k) for k in [pd.Series, pd.DataFrame] ]:
            concattable = False

        if not concattable:
            Logger.warning(f"Data {definitions} can not be concatted, keep that in mind!")
            try:
                tmpdata = pd.Series(tmpdata)
                #return tmpdata
            except:
                tmpdata = [k for k in tmpdata]
                tmpdata = pd.Series(tmpdata)
                #return tmpdata

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
        finite_data = np.isfinite(data)
        q3          = np.percentile(data[finite_data],75)
        q1          = np.percentile(data[finite_data],25)
        n_data      = len(data)
        if q3 == q1:
            Logger.warning("Can not calculate bins, falling back... to min max approach")
            q3 = max(finite_data)
            q1 = min(finite_data)

        h           = (2*(q3-q1))/(n_data**1./3)
        bins = (rightedge - leftedge)/h


        if not np.isfinite(bins):
            Logger.info(f"Got some nan somewhere: q1 : {q1}, q3 : {q3}, n_data : {n_data}, h : {h}")
            Logger.warning("Calculate Freedman-Draconis bins failed, calculated nan bins, returning fallback")
            bins = fallbackbins

        if bins < minbins:
            bins = minbins
        if bins > maxbins:
            bins = maxbins
    except Exception as e:
        Logger.warning(f"Calculate Freedman-Draconis bins failed {e}")
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
    FLUXWEIGHT      = 80
    PARAMETER       = 90 # a single parameter, no array whatsoever


##############################################################

class AbstractBaseVariable(metaclass=abc.ABCMeta):
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

    def calculate_fd_bins(self, cutmask=None):
        """
        Calculate a reasonable binning

        Keyword Args:
            cutmask (numpy.ndarray) : a boolean mask to cut on, in case 
                                      cuts have been applied to the 
                                      category this data is part of

        Returns:
            numpy.ndarray: Freedman Diaconis bins
        """
        tmpdata = self.data
        if cutmask is not None:
            if len(cutmask) > 0:
                tmpdata = tmpdata[cutmask]

        nbins = freedman_diaconis_bins(tmpdata, min(tmpdata), max(tmpdata))
        bins = np.linspace(min(tmpdata),max(tmpdata), nbins)
        return bins

    def harvest(self, *files):
        """
        Hook to the harvest method. Don't use in case of multiprocessing!
        Args:
            *files: walk through these files and readout
        """
        if self.role == VariableRole.PARAMETER:
            self._data = harvest(files, self.definitions, transformation= self.transform)
            self._data = self._data[0]
        else:
            self._data = harvest(files, self.definitions)
        self.declare_harvested()

    @abc.abstractmethod
    def rewire_variables(self, vardict):
        return

    @property
    def ndim(self):
        if hasattr(self._data, "multidim"):
            if self._data.multidim == True:
                return 2
        return self._data.ndim

    @property
    def data(self):
        if isinstance(self._data, pd.DataFrame):
            #return self._data.as_matrix()
            #FIXME: as_matrix is depracted in favor of values
            return self._data.values
        if not hasattr(self._data, "shape"):
            Logger.warning("Something's wrong, this should be array data!")
            Logger.warning(f"Seeeing {type(self._data)} data")
            Logger.warning("Attempting to fix!")
            self._data = np.asarray(self._data)

        return self._data

############################################

class Variable(AbstractBaseVariable):
    """
    A hook to a single variable read out from a file
    """

    def __init__(self, name, definitions=None,\
                 bins=None, label="", transform=None,
                 role=VariableRole.SCALAR,
                 nevents=None,
                 reduce_dimension=None):
        """
        Args:
            name                                     (str) : An unique identifier

        Keyword Args:
            definitions                             (list) : table and/or column names in underlying data
            bins                           (numpy.ndarray) : used for histograms
            label                                    (str) : used for plotting and as a label in tables
            transform                               (func) : apply to each member of the underlying data at readout
            role (HErmes.selection.variables.VariableRole) : The role the variable is playing. 
                                                             In most cases the default is the best choice
            nevents                                  (int) : number of events to read in (ROOT only right now!)
            reduce_dimension                         (int) : in case of multidimensionality,
                                                             take only the the given index of the array (ROOT only right now)
        """
        AbstractBaseVariable.__init__(self)

        if definitions is not None:
            #assert not (False in [len(x) <= 2 for x in definitions]), "Can not understand variable definitions {}!".format(definitions)
            self.defsize = len(definitions[0])
            #FIXME : not sure how important this is right now
            #assert not (False in [len(x) == self.defsize for x in definitions]), "All definitions must have the same length!"
        else:
            self.defsize = 0

        self.name        = name
        self.role        = role
        self.bins        = bins # when histogrammed
        self.label       = label
        self.transform   = transform
        self.definitions = definitions
        self._data       = pd.Series(dtype=np.float64)
        self.nevents     = nevents
        self.reduce_dimension = reduce_dimension
        self._role = role

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
                 role=VariableRole.SCALAR,
                 dtype=np.float64):
        """
        A compound variable is a variable which is created from two or more other variables. This variable does not have
        a direct representation in a file, but gets calculated on the fly instead, e.g. a residual of two other variables
        The 'operation' denoted function here defines what operator should be applied to the variables to create the new
        coumpound variable

        Args:
            name                                     (str) : An unique identifier for the new variable.

        Keyword Args:
            variables                               (list) : A list of variables used to calculate the new variable.
            label                                    (str) : A label for plotting.
            bins                              (np.ndarray) : binning for distributions.
            operation                                (fnc) : The operation which will be applied to variables.
            role (HErmes.selection.variables.VariableRole) : The role the variable is playing.
                                                             In most cases the default is the best choice. Assigning roles
                                                             to variables allows for special magic, e.g. in the case
                                                             of weighting
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
        self._data = pd.Series(dtype=np.float64) #dtype to suppress warning
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



