
import numpy as n
import variable_defs

import utils.files as f
from variables.variables import GetVariablesFromModule
from variables.variables import FreedmanDiaconisBins

datafiles = f.harvest_files("/home/achim/scripts/devel/testdata/",prefix="",ending=".h5",sanitizer=lambda x : "signal" in x)
rootfiles = f.harvest_files("/home/achim/scripts/devel/testdata/",prefix="",ending=".root")
variables = GetVariablesFromModule(variable_defs)

print datafiles
print rootfiles
print variables
allfiles = datafiles + rootfiles
for var in variables:
    #var.harvest(datafiles[0])
    var.harvest(*allfiles)
    print var,var.data

