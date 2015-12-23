"""
An interface to icecubes weighting schmagoigl
"""

from icecube import icetray,dataclasses
from icecube.weighting.weighting import from_simprod, EnergyWeight,ParticleType
from icecube.weighting.fluxes import *

ALLOWED_FLUXES = {"gaisserH3a" : GaisserH3a}


###########################################

def GetGenerator(datasets):
    """
    datasets must be a dict of dataset_id : number_of_files
    """

    generators = []
    for k in datasets.keys():
        nfiles = datasets[k]
        generator = from_simprod(k)
        generators.append(nfiles*generator)

    generator = reduce(lambda x,y : x+y, generators)
    return generator

#####################################

def HowManyFilesDB(dataset):
    """
    What does the DB think how many files 
    should be there
    """
    nfiles,__ = from_simprod(dataset)
    return nfiles

##########d##################################

def GetCorsikaWeight(model,datasets,mc_prim,mc_type):
    """
    Compute weights for CORSIKA datasets    
    """
    flux = ALLOWED_FLUXES[model]
    flux = flux()   
    gen  = GetGenerator(datasets)
    weight = EnergyWeight(gen,flux)
    return weight(mc_prim,mc_type)

###############################################

def PrintFluxes():
    return "".join(ALLOWED_FLUXES.keys())


