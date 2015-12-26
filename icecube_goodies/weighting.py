"""
An interface to icecubes weighting schmagoigl
"""

from icecube import icetray,dataclasses,NewNuFlux
from icecube.weighting.weighting import from_simprod, EnergyWeight,ParticleType
from icecube.weighting.fluxes import *

from conversions import ParticleType
import inspect

import numpy as n

FLUXES = {"gaisserH3a" : GaisserH3a}

###########################################

# safely boilerplate our own weight class,
# like in weighting, not sure if tt 
# is entirely kosher there 

class Weight(object):

    def __init__(self,generator,flux):
        self.gen = generator
        self.flux = flux

    def __call__(self,energy,ptype,zenith=[],mapping=False):

        #type mapping
        if mapping:
            pmap = {14:ParticleType.PPlus, 402:ParticleType.He4Nucleus, 1407:ParticleType.N14Nucleus, 2713:ParticleType.Al27Nucleus, 5626:ParticleType.Fe56Nucleus}
            ptype = map(lambda x : pmap[x], ptype )
        if len(zenith) > 0:
            return self.flux(energy,ptype,zenith)/self.gen(energy,particle_type=ptype,cos_theta=zenith)
        else:
            return self.flux(energy,ptype)/self.gen(energy,particle_type=ptype)

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

def GetModelWeight(model,datasets,mc_p_energy=[],mc_p_type=[],mc_p_zenith=[],model_kwargs={}):
    """
    Compute weights for CORSIKA datasets    
    """
    if callable(model):
        if not model_kwargs:
            flux = model()
        else:
            flux = model(**model_kwargs)

    else:
        if model_kwargs:
            print "Can't use model kwargs with predefined model!"
        flux = FLUXES[model]
        flux = flux()   

    gen  = GetGenerator(datasets)
    weight = Weight(gen,flux)
    return weight(mc_p_energy,mc_p_type,zenith=mc_p_zenith,mapping=True)

###############################################

def PowerLawFlux(fluxconst=1e-8,gamma=2):
    if gamma > 0:
        gamma *= -1

    # closure:)
    def flux(mc_p_energy,mc_p_type):
        # weighting API requires second argument even if we
        # don't need it
        flux = fluxconst * n.power(mc_p_energy, gamma)
        return flux
    return flux 

###############################################

def AtmosphericFlux(model='honda2006',fluxconst=1.):
    nuwflux = NewNuFlux.makeFlux(model)
    def flux(mc_p_energy,mc_p_type,mc_p_zenith):
        mc_p_type = n.int32(mc_p_type)
        mc_p_type = map(dataclasses.I3Particle.ParticleType,mc_p_type)
        return fluxconst*nuwflux.getFlux(mc_p_type,mc_p_energy,mc_p_zenith)

    return flux
      
##############################################

def PrintFluxes():
    return "".join(FLUXES.keys())


