"""
Categories of data, like "signal" of "background" etc
"""

from pyevsel.utils.files import harvest_files,DS_ID,EXP_RUN_ID
from pyevsel.utils.logger import Logger
import variables
import pandas as pd
import inspect
import numpy as n

from copy import deepcopy as copy

MC_P_EN   = "mc_p_en"
MC_P_TY   = "mc_p_ty"
MC_P_ZE   = "mc_p_ze"
RUN_START = "run_start_mjd"
RUN_STOP  = "run_stop_mjd"

class Category(object):
    """
    An interface to variables from a certain 
    type of file
    """

    
    def __init__(self,name,label="",plotcolor=''):
        """
        Args:
            name (str): a descriptive, unique name

        Keyword Args:
            label (str): a label used for plotting
        """
        self.name = name
        self.label = label
        self.datasets = dict()
        self.files = []
        try:
            self.vardict = {}
        except AttributeError:
            pass #This happens for our ReweightedSimulation class
        self._is_harvested = False
        self.weights     = pd.Series()
        self._weightfunction = None

    @staticmethod
    def _ds_regexp(filename):
        """
        A container for matching a dataset number against a filename

        Args:
            filename (str): An filename of a datafile
        Returns:
            dataset (int): A dataset number extracted from the filename
        """
        return DS_ID(filename)

    def set_weightfunction(self,func):
        """
        Register a function used for weighting
        
        Args:
            func (func): the function to be used
        """
        self._weightfunction = func


    def load_vardefs(self,module):
        """
        Load the variable definitions from a module

        Args:
            module (python module): Needs to contain variable definitions
        """

        def cleaner(x):
            try:
                return not(x.__name__.startswith("_"))
            except:
                return False
        
        all_vars = inspect.getmembers(module)#,cleaner)
        all_vars = [x[1] for x in all_vars if isinstance(x[1],variables.Variable)]
        for v in all_vars:
            if self.vardict.has_key(v.name):
                print "Variable %s already defined,skipping!" %v.name
                continue
            new_v = copy(v)
            self.vardict[new_v.name] = new_v

    def add_variable(self,variable):
        """
        Add a variable to this category

        Args: 
            variable (pyevsel.variables.variables.Variable): A Variable instalce
        """
        thisvar = copy(variable)
        self.vardict[thisvar.name] = thisvar

    def get_files(self,*args,**kwargs):
        """
        Load files for this category
        uses pyevsel.utils.files.harvest_files
        
        Args:
            *args (list of strings): Path to possible files
        
        Keyword Args:
            datasets (dict(dataset_id : nfiles)): i given, load only files from dataset dataset_id  set nfiles parameter to amount of L2 files the loaded files will represent
            force (bool): forcibly reload filelist (pre-readout vars will be lost)
            all other kwargs will be passed to
            utils.files.harvest_files
        """
        force = False
        if kwargs.has_key("force"):
            force = kwargs.pop("force")
        if self._is_harvested:
            print "Variable has already be harvested! If you really want to reload the filelist, use 'force=True'. If you do so, all your harvested variables will be deleted!"
            if not force:
                return
            else:
                print "..using force.."
       
        if kwargs.has_key("datasets"):
            filtered_files = []
            self.datasets = kwargs.pop("datasets")
            files = harvest_files(*args,**kwargs)        
            datasets = [self._ds_regexp(x) for x in files] 
            
            assert len(datasets) == len(files)
            
            ds_files = zip(datasets,files)
            for k in self.datasets.keys():
                filtered_files.extend([x[1] for x in ds_files if x[0] == k])

            files = filtered_files
            del filtered_files
        else:
            files = harvest_files(*args,**kwargs)

        self.files = files
        del files

    def get(self,varkey):
        """        
        Retrieve the data of a variable
        
        Args:
            varkey (str): The name of the variable
        
        """

        if not self.vardict.has_key(varkey):
            raise KeyError("%s not found!" %varkey)

        return self.vardict[varkey].data

    def read_variables(self,names=[]):
        """
        Harvest the variables in self.vardict

        Keyword Args:
            names (list): if != [], havest variables these variables
        """    
    
        #assert len([x for x in varlist if isinstance(x,variables.Variable)]) == len(varlist), "All variables must be instances of variables.Variable!"
        if not names:
            names = self.vardict.keys()
        compound_variables = [] #harvest them later        
        for varname in names:
            try:
                if isinstance(self.vardict[varname],variables.CompoundVariable):
                    compound_variables.append(varname)
                    continue
            except KeyError:
                print "Cannot find %s in variables!" %varname
                continue
            self.vardict[varname].harvest(*self.files)

        for varname in compound_variables:
            #FIXME check if this causes a memory leak
            self.vardict[varname]._rewire_variables(self.vardict)
            self.vardict[varname].harvest()
    
    def get_weights(self):
        raise NotImplementedError("Not implemented for base class!")

    def get_datacube(self,variablenames=[]):
        pass


    def __radd__(self,other):
        
        for k in other.datasets.keys():
            self.datasets.update({k : other.datasets[k]})
        self.files.extend(other.files)

    def __repr__(self):
        return """<Category: Category %s>""" %self.name

    def __hash__(self):
        return hash((self.name,"".join(map(str,self.datasets.keys()))))

    def __eq__(self,other):
        if (self.name == other.name) and (self.datasets.keys() == other.datasets.keys()):
            return True
        else:
            return False

    def __len__(self):
        """
        Return the longest variable element
        FIXME: introduce check?
        """    
        
        lengths = n.array([len(self.vardict[v].data) for v in self.vardict.keys()])
        lengths = lengths[lengths > 0]
        selflen = list(set(lengths))
        assert len(selflen) == 1, "Different variable lengths!"
        return selflen[0]


class Simulation(Category):

    def __init__(*args,**kwargs):
        Category.__init__(*args,**kwargs)    
        self = args[0]
        self._mc_p_set     = False
        self._mc_p_readout = False

    def __repr__(self):
        return """<Category: Simulation %s>""" %self.name
    
    def set_mc_primary(self,energy_var=variables.Variable(None),type_var=variables.Variable(None),zenith_var=variables.Variable(None)):
        """
        Let the simulation category know which 
        are the paramters describing the primary

        Keyword Args:
            energy_var (pyevself.variables.variables.Variable): simulated primary energy
            type_var (pyevself.variables.variables.Variable): simulated primary type
            zenith_var (pyevself.variables.variables.Variable): simulated primary zenith
        """
        for var,name in [(energy_var,MC_P_EN),(type_var,MC_P_TY),(zenith_var,MC_P_ZE)]:
            if var.name is None:
                print "No %s available" %name
            elif self.vardict.has_key(name):
                print "..%s already defined, skipping..."
                continue
            
            else:
                if var.name != name:
                    print "..renaming %s to %s.." %(var.name,name)        
                    var.name = name
                newvar = copy(var)
                self.vardict[name] = newvar

        self._mc_p_set = True

    def read_mc_primary(self):
        """
        Trigger the readout of MC Primary information
        """
        if not self._mc_p_set:
            raise ValueError("Variable for MC Primary are not defined! Run set_m_primary first!")
        self.read_variables([MC_P_EN,MC_P_TY,MC_P_ZE])
        self._mc_p_readout = True

    #def read_variables(self,names=[])
    #    super(Simulation,self).read_variable(names)
    #    for k in self.vardict.keys()

    def get_weights(self,model,model_kwargs = {}):
        """
        Calculate weights for the variables in this category

        Args:
            model (callable): A model to be evaluated

        Keyword Args:
            model_kwargs (dict): Will be passed to model
        """
        if not self._mc_p_readout:
            self.read_mc_primary()

        func_kwargs = {"mc_p_energy" : self.get(MC_P_EN),\
                       "mc_p_type" :self.get(MC_P_TY)}

        #for key in func_kwargs.keys():
        #    if not key in self._weightfunction.func_code.co_varnames:
        #        func_kwargs.pop(key)

        try:
            func_kwargs["mc_p_zenith"] = self.get(MC_P_ZE)
        except KeyError:
            print "No MCPrimary zenith informatiion! Trying to omit.."

        func_kwargs.update(model_kwargs)

        self.weights = pd.Series(self._weightfunction(model,self.datasets,\
                                 **func_kwargs))
                                 #self.vardict[mc_p_energy].data,\
                                 #self.vardict[mc_p_type].data))

        @property
        def livetime(self):
            return self.weights.sum() / n.power(self.weights, 2).sum()



class ReweightedSimulation(Simulation):
    """
    A proxy for simulation dataset, when only the weighting differs
    """

    def __init__(self,name,mother,label="",plotcolor="red"):
        self._mother = mother
        self.name = name
        self.plotcolor = plotcolor
        self.label = label
        self.weights     = pd.Series()
        self._weightfunction = None

    #proxy the stuff by hand
    #FIXME: there must be a better way

    @property
    def vardict(self):
        return self._mother.vardict

    #def __getattr__(self,attr):
    #    self._mother.__getattribute__(self,attr)

    @property
    def datasets(self):
        return self._mother.datasets

    @property
    def files(self):
        return self._mother.files

    @property
    def _is_harvested(self):
        return self._mother._is_harvested

    @property
    def _mc_p_set(self):
        return self._mother._mc_p_set

    @property
    def _mc_p_readout(self):
        return self._mother._mc_p_readout 


    def read_variables(self,names=[]):
        raise NotImplementedError("Use read_variables of the mother categorz")

    def __radd__(self,other):
        raise NotImplementedError
 
    def set_mc_primary(self,energy_var=variables.Variable(None),type_var=variables.Variable(None),zenith_var=variables.Variable(None)):
        raise NotImplementedError

    def read_mc_primary(self):
        raise NotImplementedError




    def __repr__(self):
        return """<Category: ReweightedSimulation %s from %s>""" %(self.name,self._mother)


class Data(Category):

    def __init__(self,*args,**kwargs):
        print "Runs are considered as datasets..."
        Category.__init__(self,*args,**kwargs)    
        self._livetime = 0
        self._runstartstop_set = False

    @staticmethod
    def _ds_regexp(filename):
        return EXP_RUN_ID(filename)

    def set_livetime(self,livetime):
        self._livetime = livetime

    # livetime is read-only
    @property
    def livetime(self):
        return self._livetime


    def set_run_start_stop(self,runstart_var=variables.Variable(None),runstop_var=variables.Variable(None)):
        """
        Let the simulation category know which 
        are the paramters describing the primary

        Keyword Args:
            runstart_var (pyevself.variables.variables.Variable): beginning of a run
            runstop_var (pyevself.variables.variables.Variable): beginning of a run

        """
        #FIXME
        for var,name in [(runstart_var,RUN_START),(runstop_var,RUN_STOP)]:
            if var.name is None:
                Logger.warning("No %s available" %name)
            elif self.vardict.has_key(name):
                Logger.info("..%s already defined, skipping..." %name)
                continue
            
            else:
                if var.name != name:
                    Logger.info("..renaming %s to %s.." %(var.name,name))        
                    var.name = name
                newvar = copy(var)
                self.vardict[name] = newvar

        self._runstartstop_set = True
        


    def estimate_livetime(self,force=False):
        """
        Calculate the livetime from run start/stop times, account for gaps
        
        Keyword Args:
            force (bool): overide existing livetime
        """
        if self.livetime:
            Logger.warning("There is already a livetime of %4.2f " %self.livetime)
            if force:
                Logger.warning("Applying force...")
            else:
                Logger.warning("If you really want to do this, use force = True")
                return
        
        if not self._runstartstop_set:
            if (RUN_STOP in self.vardict.keys()) and (RUN_START in self.vardict.keys()):
                self._runstartstop_set = True
            else:
                Logger.warning("Need to set run start and stop times first! use object.set_run_start_stop")
                return

        Logger.warning("This is a crude estimate! Rather use a good run list or something!")
        lengths = self.get(RUN_STOP) - self.get(RUN_START)
        gaps    = self.get(RUN_START)[1:] - self.get(RUN_STOP)[:-1] #trust me!
        #h = self.nodes["header"].read()
        #h0 = h[:-1]
        #h1 = h[1:]
        ##FIXME
        #lengths = ((h["time_end_mjd_day"] - h["time_start_mjd_day"]) * 24. * 3600. +
        #           (h["time_end_mjd_sec"] - h["time_start_mjd_sec"]) +
        #           (h["time_end_mjd_ns"] - h["time_start_mjd_ns"])*1e-9 )
 
        #gaps = ((h1["time_start_mjd_day"] - h0["time_end_mjd_day"]) * 24.  * 3600. +
        #        (h1["time_start_mjd_sec"] - h0["time_end_mjd_sec"]) +
        #        (h1["time_start_mjd_ns"] - h0["time_end_mjd_ns"])*1e-9)
 

        # detector livetime is the duration of all events + the length of      all
        # gaps between events that are short enough to be not downtime. (     guess: 30s)
        est_ltime =  ( lengths.sum() + gaps[(0<gaps) & (gaps<30)].sum() )
        self.set_livetime(est_ltime)
        return 

    def get_weights(self):
        
        # get a random variable
        for k in self.vardict.keys():
            datalen = len(self.get(k))
            if datalen:
                break
            else:
                Logger.warning("Read out variables first!")        

        self.weights = pd.Series(n.ones(datalen,dtype=n.float64)/self.livetime)


    def __repr__(self):
        return """<Category: Data %s>""" %self.name

#################################################################

class Dataset(object):
    """
    Holds many different categories
    """
    _categories = dict()

    def __init__(self,categories=[]):
        
        if categories:
            for c in categories:
                self._categories[c.name] = category
        self._weights_available = False

    def add_category(self,category):
        self._categories[category.name] = category

    @property
    def weights(self):
        pass

    def get(self,varname):
        pass

    
        

