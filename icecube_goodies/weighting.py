"""
An interface to icecubes weighting schmagoigl
"""

from icecube import icetray,dataclasses
from icecube.weighting import from_simprod, EnergyWeight,ParticleType

###########################################

def _getGenerator(datasets):
    """
    datasets must be a dict of dataset_id : number_of_files
    """

    generators = []
    for k in datasets.keys():
        nfiles = dataset[i]
        nfiles,generator = from_simprod(k)
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

def GetdCorsikaWeight(model=)
    pass

