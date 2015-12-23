"""
Categories of data, like "signal" of "background" etc
"""

from pyevsel.utils.files import harvest_files,DS_ID,EXP_RUN_ID
import variables
import pandas as pd
import inspect


from copy import deepcopy as copy

class Category(object):
    
    def __init__(self,name,label=""):
        self.name = name
        self.label = label
        self.datasets = dict()
        self.files = []
        self.vardict = {}
        self._is_harvested = False
        self.weights     = pd.Series()
        self._weightfunction = None

    @staticmethod
    def _ds_regexp(filename):
        return DS_ID(filename)

    def set_weightfunction(self,func):
        self._weightfunction = func


    def load_vardefs(self,module):
        """
        Load the variable definitions from a module
        """

        def cleaner(x):
            try:
                return not(x.__name__.startswith("_"))
            except:
                return False
        
        all_vars = inspect.getmembers(module)#,cleaner)
        all_vars = [x[1] for x in all_vars if isinstance(x[1],variables.Variable)]
        for v in all_vars:
            new_v = copy(v)
            self.vardict[new_v.name] = new_v

    def add_variable(self,variable):
        thisvar = copy(variable)
        self.vardict[thisvar.name] = thisvar

    def get_files(self,*args,**kwargs):
        """
        get_files(path,force=False,**kwargs)
        load files for this datatype with
        utils.files.harvest_files
        :datasets = dict(dataset_id : nfiles): 
                if given, load only files from dataset dataset_id
                set nfiles parameter to amount of L2 files
                the loaded files will represent
        :force = True|False: forcible reload filelist, even
                if variables have been read out
                already
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
        Shortcut for quick data access
        """

        if not self.vardict.has_key(varkey):
            raise KeyError("%s not found!" %varkey)

        return self.vardict[varkey].data

    def read_variables(self,names=[]):
        """
        Harvest the variables in self.vardict
        :names: only variables with this names 
                will be harvested
        """    
    
        #assert len([x for x in varlist if isinstance(x,variables.Variable)]) == len(varlist), "All variables must be instances of variables.Variable!"
        if not names:
            names = self.vardict.keys()
        
        for varname in names:
        #for var in varlist:
            self.vardict[varname].harvest(*self.files)
            #var.harvest(*self.files)
            #self.vardict[var.name] = var
            
        #del var
    
    def get_weights(self,model,mc_p_energy="mc_p_en",mc_p_type="mc_p_ty"):
        """
        Calculate weights for the
        variables in this category
        """
        self.weights = pd.Series(self._weightfunction(model,self.datasets,\
                                 self.vardict[mc_p_energy].data,\
                                 self.vardict[mc_p_type].data))


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

    

class Signal(Category):

   def __repr__(self):
        return """<Category: Signal %s>""" %self.name

class Background(Category):

       def __repr__(self):
        return """<Category: Background %s>""" %self.name

class Data(Category):

    def __init__(self,name,label=""):
        print "Runs are considered as datasets..."
        Category.__init__(self,name,label=label)    

    @staticmethod
    def _ds_regexp(filename):
        return EXP_RUN_ID(filename)

    def __repr__(self):
        return """<Category: Data %s>""" %self.name

# needs enum34
#
#from enum import Enum
#
#class Categories(Enum):
#
#    def __init__(self,*args,**kwargs):
#        Enum.__init__(self,*args,**kwargs)
#        self.signaltypes = []
#        self.backgroundtypes = []
#
#
#    def define_signal(self,*args):
#        self.signaltypes = args
#
#    def get_signal(self):
#        for i in self.signaltypes:
#            pass            
#
#if __name__ == "__main__":
#
#    data = Categories("Cats","nue numu mu")
#    print data
#    print data.nue
#
#
