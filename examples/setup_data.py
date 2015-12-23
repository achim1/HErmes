
import numpy as n
import variable_defs

import pyevsel.utils.files as f
from pyevsel.variables.variables import GetVariablesFromModule
from pyevsel.variables.variables import FreedmanDiaconisBins
from pyevsel.variables.categories import *

signal = Signal("nue",label=r"$\nu_e$")
signal.get_files("/home/achim/scripts/devel/testdata/",prefix="",ending=".h5",sanitizer=lambda x : "signal" in x)

rootfiles = f.harvest_files("/home/achim/scripts/devel/testdata/",prefix="",ending=".root")
signal.files.extend(rootfiles)
print signal
#datafiles = f.harvest_files("/home/achim/scripts/devel/testdata/",prefix="",ending=".h5",sanitizer=lambda x : "signal" in x)
variables = GetVariablesFromModule(variable_defs)
signal.read_variables(variables)
#print datafiles
#print rootfiles
print variables
#allfiles = datafiles + rootfiles
vardict = {}
for var in variables:
    #var.harvest(datafiles[0])
    var.harvest(*signal.files)
vardict[signal] = var
print vardict 
