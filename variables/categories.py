"""
Categories of data, like "signal" of "background" etc
"""

from utils.files import harvest_files
import variables

class Category(object):
    
    def __init__(self,name,label="",dataset=1):
        self.name = name
        self.label = label
        self.dataset = [dataset]
        self.files = []
        self.vardict = {}
        self.weights = []
        self._is_harvested = False
    
    def get_files(self,*args,**kwargs):
        """
        get_files(path,force=False,**kwargs)
        load files for this datatype with
        utils.files.harvest_files
        :force: forcible reload filelist, even
                if variables have been read out
                already
        all other kwargs will be passed to
        utils.files.harvest_files
        """
        force = False
        if kwargs.has_key("force"):
            force = kwargs["force"]
        if self._is_harvested:
            print "Variable has already be harvested! If you really want to reload the filelist, use 'force=True'. If you do so, all your harvested variables will be deleted!"
            if not force:
                return
            else:
                print "..using force.."
        
        self.files = harvest_files(*args,**kwargs)        
        del self.vardict
        self.vardict = {}    

    def read_variables(self,varlist):
        
        assert len([x for x in varlist if isinstance(x,variables.Variable)]) == len(varlist), "All variables must be instances of variables.Variable!"

        for var in varlist:
            var.harvest(*self.files)
            self.vardict[var.name] = var
            
        del var
    
    def get_weights(self):
        raise NotImplementedError


    def __radd__(self,other):
        
        self.dataset.append(other.dataset)
        self.files.extend(other.files)

    def __repr__(self):
        return """<Category: Category %s>""" %self.name

    def __hash__(self):
        return hash((self.name,"".join(map(str,self.dataset))))

    def __eq__(self,other):
        if (self.name == other.name) and (self.dataset == other.dataset):
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
