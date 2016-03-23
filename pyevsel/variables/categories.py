"""
Categories of data, like "signal" of "background" etc
"""

from pyevsel.utils.files import harvest_files,DS_ID,EXP_RUN_ID
from pyevsel.utils.logger import Logger
from pyevsel.utils import GetTiming
from pyevsel.plotting.plotting import VariableDistributionPlot
from pyevsel.plotting import GetCategoryConfig

import variables
import pandas as pd
import inspect
import numpy as n

from dashi.tinytable import TinyTable

from copy import deepcopy as copy

MC_P_EN   = "mc_p_en"
MC_P_TY   = "mc_p_ty"
MC_P_ZE   = "mc_p_ze"
RUN_START = "run_start_mjd"
RUN_STOP  = "run_stop_mjd"
RUN       = "run"
EVENT     = "event"

class Category(object):
    """
    An interface to variables from a certain 
    type of file
    """

    def __init__(self,name):
        """
        Args:
            name (str): a descriptive, unique name
        """
        self.name = name
        self.datasets = dict()
        self.files = []
        self.cuts  = []
        self.cutmask = n.array([])

        try:
            self.vardict = {}
        except AttributeError:
            pass #This happens for our ReweightedSimulation class
        self._weights     = pd.Series()
        self._weightfunction = None

        self._is_harvested = False
        self._raw_count = 0 #how much 'raw events' (unweighted)

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

    def _get_raw_count(self):
        """
        Calculate how many raw events are in this
        category

        """

        assert self._is_harvested,"Please read out variables first"
        self._raw_count = len(self.get(RUN))

    # read only
    @property
    def raw_count(self):
        return self._raw_count

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
                Logger.debug("Variable %s already defined,skipping!" %v.name)
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
            Logger.info("Variables have already been harvested!if you really want to reload the filelist, use 'force=True'. If you do so, all your harvested variables will be deleted!")
            if not force:
                return
            else:
                Logger.warning("..using force..")
       
        if "datasets" in kwargs:
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

    def get(self,varkey,uncut=False):
        """        
        Retrieve the data of a variable
        
        Args:
            varkey (str): The name of the variable

        Keyword Args:
            uncut (bool): never return cutted values
        """

        if varkey not in self.vardict:
            raise KeyError("%s not found!" %varkey)

        if len(self.cutmask) and not uncut:
            return self.vardict[varkey].data[self.cutmask]
        else:
            return self.vardict[varkey].data

    @GetTiming
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

                if isinstance(self.vardict[varname],variables.VariableList):
                    compound_variables.append(varname)
                    continue
            except KeyError:
                Logger.info("Cannot find %s in variables!" %varname)
                continue
            self.vardict[varname].harvest(*self.files)

        for varname in compound_variables:
            #FIXME check if this causes a memory leak
            self.vardict[varname]._rewire_variables(self.vardict)
            self.vardict[varname].harvest()
        self._is_harvested = True
        self._get_raw_count()

    def get_weights(self):
        raise NotImplementedError("Not implemented for base class!")

    def get_datacube(self,variablenames=[]):
        cube = dict()
        for k in self.vardict.keys():
            cube[k] = self.get(k)

        return pd.DataFrame(cube)

    @property
    def weights(self):
        if len(self.cutmask):
            return self._weights[self.cutmask]
        else:
            return self._weights

    @property
    def variablenames(self):
        return self.vardict.keys()

    def add_cut(self,cut):
        """
        Add a cut without applying it yet

        Args:
            cut (pyevsel.variables.cut.Cut): Append this cut to the internal cutlist

        """

        self.cuts.append(cut)

    def apply_cuts(self,inplace=False):
        """
        Apply the added cuts

        Keyword Args:
            inplace (bool): If True, cut the internal variable buffer
                           (Can not be undone except variable is reloaded)
        """
        self.cutmask = n.array([])
        mask = n.ones(self.raw_count)
        for cut in self.cuts:
            for varname,cutfunc in cut:
                mask = n.logical_and(mask,self.get(varname).apply(cutfunc))
                if not sum(mask):
                    Logger.warning("After cutting on %s, no events are left!" %varname)
            if cut.condition is not None:
                if n.logical_or(mask,n.logical_not(cut.condition[self.name])):
                    #FIXME
                    mask = n.logical_and(cut.condition,mask)
        if inplace:
            for k in self.vardict.keys():
                self.vardict[k].data = self.vardict[k].data[mask]
        else:
            self.cutmask = n.array(mask)

        return

    def undo_cuts(self):
        """
        Conveniently undo a previous "apply_cuts"
        """

        self.cutmask = n.array([])

    def delete_cuts(self):
        """
        Get rid of previously added cuts
        """

        self.undo_cuts()
        for cut in self.cuts:
            del cut #FIXME: explicit call to destructor
                    # should not be necessary
        self.cuts = []

    @property
    def integrated_rate(self):
        """
        Calculate the total eventrate of this category
        (requires weights)

        Returns (tuple): rate and quadratic error
        """

        rate  = self.weights.sum()
        error = n.sqrt((self.weights**2).sum())

        return (rate,error)


    def add_livetime_weighted(self,other,self_livetime=None,other_livetime=None):
        """
        Combine two datasets livetime weighted. If it is simulated data,
        then in general it does not know about the detector livetime.
        In this case the livetimes for the two datasets can be given

        Args:
            other (pyevsel.categories.Category): Add this dataset

        Keyword Args:
            self_livetime (float): the data livetime for this dataset
            other_livetime (float): the data livetime for the other dataset

        """

        assert self.vardict.keys() == other.vardict.keys(),"Must have the same variables to be combined"

        if isinstance(self,Data):
            self_livetime = self.livetime

        if isinstance(other,Data):
            other_livetime = other.livetime

        for k in other.datasets.keys():
            self.datasets.update({k : other.datasets[k]})
        self.files.extend(other.files)
        if self.cuts or other.cuts:
            self.cuts.extend(other.cuts)
        if len(self.cutmask) or len(other.cutmask):
            self.cutmask = n.hstack((self.cutmask,other.cutmask))

        for name in self.variablenames:
            self.vardict[name].data = pd.concat([self.vardict[name].data,other.vardict[name].data])

        self_weight = (self_livetime/(self_livetime + other_livetime))
        other_weight = (other_livetime/(self_livetime + other_livetime))

        self._weights = pd.concat([self_weight*self._weights,other_weight*other._weights])
        if isinstance(self,Data):
            self.set_livetime(self.livetime + other.livetime)
        self._get_raw_count()

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
    """
    An interface to variables from simulated data
    Allows to weight the events
    """

    def __init__(self,name):
        Category.__init__(self,name)
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
                Logger.warning("No %s available" %name)
            elif name in self.vardict:
                Logger.warning("..%s already defined, skipping...")
                continue
            
            else:
                if var.name != name:
                    Logger.warning("..renaming %s to %s.." %(var.name,name))
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
            Logger.warning("No MCPrimary zenith informatiion! Trying to omit..")

        func_kwargs.update(model_kwargs)

        self._weights = pd.Series(self._weightfunction(model,self.datasets,\
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

    def __init__(self,name,mother):
        Simulation.__init__(self,name)
        self._mother = mother
        self.name = name
        self._weights = pd.Series()
        self._weightfunction = None
        self._raw_count = self._mother.raw_count

    #proxy the stuff by hand
    #FIXME: there must be a better way

    @property
    def vardict(self):
        return self._mother.vardict

    def get_datasets(self):
        return self._mother.datasets

    def set_datasets(self,datasets):
        pass

    def get_files(self):
        return self._mother.files

    def set_files(self,files):
        pass

    def get_is_harvested(self):
        return self._mother._is_harvested

    def set_is_harvested(self,value):
        pass

    def get_mc_p_set(self):
        return self._mother._mc_p_set

    def set_mc_p_set(self,value):
        pass

    def get_mc_p_readout(self):
        return self._mother._mc_p_readout

    def set_mc_p_readout(self,value):
        pass

    datasets = property(get_datasets,set_datasets)
    files = property(get_files,set_files)
    _is_harvested = property(get_is_harvested,set_is_harvested)
    _mc_p_readout = property(get_mc_p_readout,set_mc_p_readout)

    def read_variables(self,names=[]):
        Logger.warning("Use read_variables of the mother category. Not doing anything...")

    def __radd__(self,other):
        raise NotImplementedError
 
    def set_mc_primary(self,energy_var=variables.Variable(None),type_var=variables.Variable(None),zenith_var=variables.Variable(None)):
        return self._mother.set_mc_primary(energy_var=energy_var,type_var=type_var,zenith_var=zenith_var)

    def read_mc_primary(self):
        raise NotImplementedError

    def __repr__(self):
        return """<Category: ReweightedSimulation %s from %s>""" %(self.name,self._mother)

    def add_livetime_weighted(self,other):
        raise ValueError('ReweightedSimulation datasets can not be combined! Instanciate after adding mothers instead!')

    def get(self,varkey,uncut=False):
        data = self._mother.get(varkey,uncut=True)

        if len(self.cutmask) and not uncut:
            return data[self.cutmask]
        else:
            return data
    
    @property
    def raw_count(self):
        return self._mother.raw_count

class Data(Category):
    """
    An interface to real time event data
    Simplified weighting only
    """

    def __init__(self,name,livetime=0):
        """
        Instanciate a Data dataset. Provide livetime in **kwargs.
        Special keyword "guess" for livetime allows to guess the livetime later on

        Args:
            *args:
            **kwargs:

        Returns:

        """
        Category.__init__(self,name)
        self.set_livetime(livetime)
        self._runstartstop_set = False

    @staticmethod
    def _ds_regexp(filename):
        return EXP_RUN_ID(filename)

    def set_livetime(self,livetime):
        """
        Override the private _livetime member

        Args:
            livetime: The time needed for data-taking

        Returns: None

        """

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
        if self.livetime and (not self.livetime=="guess"):
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

    def set_weightfunction(self,func):
        """
        Can not use this for data, override...

        Args:
            func:

        Returns:

        """
        pass

    def get_weights(self):
        """
        Calculate weights as rate, that is number of
        events per livetime
        """
        if self.livetime == "guess":
            self.estimate_livetime()
        self._weights = pd.Series(n.ones(self.raw_count,dtype=n.float64)/self.livetime)

    def __repr__(self):
        return """<Category: Data %s>""" %self.name

#################################################################

class Dataset(object):
    """
    Holds different categories, relays calls to each
    of them
    """
    categories = []
    livetime   = 0

    def __init__(self,*args):
        """
        Iniitalize with the categories

        Args:
            *args: pyevsel.variables.categories.Category list

        Returns:

        """
        if args:
            for cat in args:
                self.categories.append(cat)
                self.__dict__[cat.name] = cat

    def read_all_vars(self,variable_defs):
        """
        Read out the variable for all categories

        Args:
            variable_defs: A python module containing variable definitions

        Returns:

        """
        for cat in self.categories:
            if isinstance(cat,Simulation):
                cat.set_mc_primary(energy_var=variable_defs.mc_p_en,\
                                   type_var=variable_defs.mc_p_ty,\
                                   zenith_var=variable_defs.mc_p_ze)
            cat.load_vardefs(variable_defs)
            cat.read_variables()

    def set_weightfunction(self,weightfunction=lambda x:x):
        """

        Args:
            weightfunction (func or dict): if func is provided, set this to all categories
                                           if needed, provide dict, cat.name -> func for individula setting

        Returns: None

        """
        if isinstance(weightfunction,dict):
            for cat in self.categories:
                cat.set_weightfunction(weightfunction[cat.name])

        else:
            for cat in self.categories:
                cat.set_weightfunction(weightfunction)

    def get_weights(self,models):
        """
        Calculate the weights for all categories

        Args:
            weightfunction (func): set func used for medel weight calculation
            models (dict): A dictionary of categoryname -> model
        """
        for cat in self.categories:
            if not cat.name in models:
                cat.get_weights()
            else:
                cat.get_weights(models[cat.name])

    def add_category(self,category):
        """
        Add another category to the dataset

        Args:
            category (pyevsel.categories.Category): add this category

        """

        self.categories.append(category)

    def get_category(self,categoryname):
        """
        Get a reference to a category

        Args:
            category: A name which has to be associated to a category

        Returns (pyevsel.variables.categories.Category): Category
        """

        for cat in self.categories:
            if cat.name == categoryname:
                return cat

        raise KeyError("Can not find category %s" %categoryname)

    def get_variable(self,varname):
        """
        Get a pandas dataframe for all categories

        Args:
            varname (str): A name of a variable

        Returns (pandas.DataFrame): A 2d dataframe category -> variable
        """

        var = dict()
        for cat in self.categories:
            var[cat.name] = cat.get(varname)

        df = pd.DataFrame(var)
        return df

    @property
    def weights(self):
        w = dict()
        for cat in self.categories:
            w[cat.name] = cat.weights
        df = pd.DataFrame(w)
        return df

    def __repr__(self):
        """
        String representation
        """

        rep = """ <Dataset: """
        for cat in self.categories:
            rep += "%s " %cat.name
        rep += ">"

    def add_cut(self,cut):
        """
        Add a cut without applying it yet

        Args:
            cut (pyevsel.variables.cut.Cut): Append this cut to the internal cutlist

        """
        for cat in self.categories:
            cat.add_cut(cut)

    def apply_cuts(self,inplace=False):
        """
        Apply them all!
        """
        for cat in self.categories:
            cat.apply_cuts(inplace=inplace)

    def undo_cuts(self):
        """
        Undo previously done cuts, but keep them so that
        they can be re-applied
        """
        for cat in self.categories:
            cat.undo_cuts()

    def delete_cuts(self):
        """
        Completely purge all cuts from this
        dataset
        """
        for cat in self.categories:
            cat.delete_cuts()

    @property
    def categorynames(self):
        return [cat.name for cat in self.categories]

    def plot_distribution(self,name,\
                          ratio=([],[]),
                          cumulative=True,
                          heights=[.4,.2,.2],
                          savepath="",savename="vdistplot"):
        """
        One shot short-cut for one of the most used
        plots in eventselections

        Args:
            name (string): The name of the variable to plot

        Keyword Args:
            ratio (list): A ratio plot of these categories will be crated
        """
        plot = VariableDistributionPlot()
        for cat in self.categories:
            plot.add_variable(cat,name)
            if cumulative:
                plot.add_cumul(cat.name)
        if len(ratio[0]) and len(ratio[1]):
            plot.add_ratio(ratio[0],ratio[1])
        plot.plot(heights=heights)
        #plot.add_legend()
        plot.canvas.save("",savename,dpi=350)
        return plot

    @property
    def integrated_rate(self):
        """
        Integrated rate for each category

        Returns (pandas.Panel): rate with error
        """
        #ratedict = dict()
        #errdict  = dict()
        rdata,edata,index = [],[],[]
        for cat in self.categories:
            rate,error = cat.integrated_rate
            rdata.append(rate)
            index.append(cat.name)
            edata.append(error)

            #ratedict[cat.name] = [rate]
            #errdict[cat.name] = [errdict]
        rate = pd.Series(rdata,index)
        err  = pd.Series(edata,index)
        return (rate,err)

    def sum_rate(self,categories=[]):
        """
        Sum up the integrated rates for categories

        Args:
            background: categories considerred background

        Returns:
             tuple: rate with error

        """
        rate,error = categories[0].integrated_rate
        error = error**2
        for cat in categories[1:]:
            tmprate,tmperror = cat.integrated_rate
            rate  += tmprate # categories should be independent
            error += tmperror**2
        return (rate,n.sqrt(error))

    def _setup_table_data(self,signal=[],background=[]):
        """
        Setup data for a table
        If signal and background are given, also summed values
        will be in the list

        Keyword Args:
            signal (list): category names which are considered signal
            background (list): category names which are considered background

        Returns (dict): table dictionary
        """

        rates, errors = self.integrated_rate
        sgrate, sgerrors = self.sum_rate(signal)
        bgrate, bgerrors = self.sum_rate(background)
        allrate, allerrors = self.sum_rate(self.categories)
        tmprates  = pd.Series([sgrate,bgrate,allrate],index=["signal","background","all"])
        tmperrors = pd.Series([sgerrors,bgerrors,allerrors],index=["signal","background","all"])
        rates = rates.append(tmprates)
        errors = errors.append(tmperrors)

        datacats = []
        for cat in self.categories:
            if isinstance(cat,Data):
                datacats.append(cat)
        if datacats:
            simcats = [cat for cat in self.categories if cat.name not in [kitty.name for kitty in datacats]]
            simrate, simerror = self.sum_rate(simcats)

        fudges = dict()
        for cat in datacats:
            rate,error = cat.integrated_rate
            try:
                fudges[cat.name] = (rate/simrate,error/simerror)
            except ZeroDivisionError:
                fudges[cat.name] = n.Nan
        #table_dict["rates (evts/sec)"] = self.integrated_rate[0]
        #table_dict["stat. error (+-)"] = self.integrated_rate[1]
        rate_dict = dict()
        all_fudge_dict = dict()
        for catname in self.categorynames:
            cfg = GetCategoryConfig(catname)
            label = cfg["label"]
            rate_dict[label] = (rates[catname],errors[catname])
            if catname in fudges:
                all_fudge_dict[label] = fudges[catname]
            else:
                all_fudge_dict[label] = None

        rate_dict["Sig."] =  (rates["signal"],errors["signal"] )
        rate_dict["Bg."] = (rates["background"],errors["background"])
        rate_dict["Gr. Tot."] = (rates["all"],errors["all"])
        all_fudge_dict["Sig."]     = None
        all_fudge_dict["Bg."]      = None
        all_fudge_dict["Gr. Tot."] = None
        return rate_dict,all_fudge_dict

    def tinytable(self,signal=[],\
                    background=[],\
                    layout="v",\
                    format="html"):
        """
        Use dashi.tinytable.TinyTable to render a nice
        html representation of a rate table

        Args:
            signal (list) : summing up signal categories to calculate total signal rate
            background (list): summing up background categories to calculate total background rate
            layout (str) : "v" for vertical, "h" for horizontal
            format (str) : "html","latex","wiki"

        Returns:
            str: formatted table in desired markup
        """
        def cellformatter(input):
            #print input
            if input is None:
                return "-"
            if isinstance(input[0],pd.Series):
                input = (input[1][0],input[1][0])
            return "%4.2e +- %4.2e" %(input[0],input[1])

        rates,fudges = self._setup_table_data(signal=signal,background=background)
        tt = TinyTable()
        tt.add("Rate (1/s)", **rates)
        tt.add("Ratio",**fudges)
        return tt.render(layout=layout,format=format,format_cell=cellformatter)
        

