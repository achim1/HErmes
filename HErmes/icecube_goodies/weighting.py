"""
An interface to icecube's weighting schmagoigl
"""


try:
    from icecube.weighting.weighting import from_simprod, EnergyWeight,ParticleType
except ImportError:
    print ("WARNING: module icecube not found!")

from ..utils import Logger
from ..selection.magic_keywords import  MC_P_EN,\
                            MC_P_TY,\
                            MC_P_ZE,\
                            MC_P_WE,\
                            RUN_START,\
                            RUN_STOP, \
                            RUN,\
                            EVENT

import numpy as np
import inspect
from . import conversions as conv
from functools import reduce


NUTYPES = [conv.PDGCode.NuE,conv.PDGCode.NuEBar]
NUTYPES.extend([conv.PDGCode.NuMu,conv.PDGCode.NuMuBar])
NUTYPES.extend([conv.PDGCode.NuTau,conv.PDGCode.NuTauBar])


###########################################

# safely boilerplate our own weight class,
# like in weighting, not sure if tt 
# is entirely kosher there 

class Weight(object):
    """
    Provides the weights for 
    weighted MC simulation.
    Uses the pdf from simulation 
    and the desired flux
    """

    def __init__(self,generator,flux):
        self.gen = generator
        self.flux = flux

    def __call__(self,energy,ptype,\
                 zenith=None,mapping=False,
                 weight=None):
        """


        Args:
            energy: primary MC energy
            ptype: primary MC particle type
            zenith: cos (?) zenith

        Keyword Args:
            mapping: do a mapping to pdg
            weight: e.g. interactionprobabilityweights for nue

        Returns:
            numpy.ndarray: weights
        """

        # FIXME: mapping argument should go away
        if mapping:
            pmap = {14:ParticleType.PPlus, 402:ParticleType.He4Nucleus, 1407:ParticleType.N14Nucleus, 2713:ParticleType.Al27Nucleus, 5626:ParticleType.Fe56Nucleus}
            ptype = [pmap[x] for x in ptype]

        # FIXME: This is too ugly and not general
        can_use_zenith = False
        if hasattr(self.flux,"__call__"):
            if hasattr(self.flux.__call__,"__func__"):
                args = inspect.getargs(self.flux.__call__.__func__.__code__)
                if len(args.args) == 4: # account for self
                    can_use_zenith = True
            else:
                can_use_zenith = True # method wrapper created by NewNuflux 
        else:
            args = inspect.getargs(self.flux.__code__) 
            if len(args.args) == 3:
                can_use_zenith = True
        if (zenith is not None) and can_use_zenith:
            Logger.debug("Using zenith!")
            return self.flux(energy,ptype,zenith)/self.gen(energy,particle_type=ptype,cos_theta=zenith)
        else:
            Logger.debug("Not using zenith!")
            return self.flux(energy,ptype)/self.gen(energy,particle_type=ptype,cos_theta=zenith)

###########################################


def GetGenerator(datasets):
    """
    datasets must be a dict of dataset_id : number_of_files

    Args:
        datasets (dict): Query the database for these datasets.
                         dict dataset_id -> number of files

    Returns (icecube.weighting...): Generation probability object
    """

    generators = []
    for k in list(datasets.keys()):
        nfiles = datasets[k]
        generator = from_simprod(k)
        # depending on the version of the
        # weighting module, either nfiles,generator
        # or just generator is returned
        if isinstance(generator,tuple):
            generator = generator[1]

        generators.append(nfiles*generator)

    generator = reduce(lambda x,y : x+y, generators)
    return generator

###########################################


def constant_weights(size, scale=1.):
    """
    Calculate a constant weight for all the entries, e.g. unity

    Args:
        size (int): The size of the returned arraz (d)

    Keyword Args:
        scale (float): The returned weight is 1/scale

    Returns:
        np.ndarray
    """
    return (1/scale)*np.ones(size)


############################################


def GetModelWeight(model,datasets,\
                   mc_datasets=None,\
                   mc_p_en=None,\
                   mc_p_ty=None,\
                   mc_p_ze=None,\
                   mc_p_we=1.,\
                   mc_p_ts=1.,\
                   mc_p_gw=1.,\
                   **model_kwargs):
    """
    Compute weights using a predefined model

    Args:
        model (func): Used to calculate the target flux
        datasets (dict): Get the generation pdf for these datasets from the db
                         dict needs to be dataset_id -> nfiles
    Keyword Args:
        mc_p_en (array-like): primary energy
        mc_p_ty (array-like): primary particle type
        mc_p_ze (array-like): primary particle cos(zenith)
        mc_p_we (array-like): weight for mc primary, e.g. some interaction probability

    Returns (array-like): Weights
    """
    if model_kwargs:
        flux = model(**model_kwargs)
    else:
        flux = model()
    # FIXME: There is a factor of 5000 not accounted
    # for -> 1e4 is for the conversion of
    factor = 1.
    gen  = GetGenerator(datasets)
    if [k for k in map(int,list(gen.spectra.keys()))][0] in NUTYPES:
        Logger.debug('Patching weights')
        factor = 5000
    weight = Weight(gen,flux)
    return factor*mc_p_we*weight(mc_p_en,mc_p_ty,zenith=mc_p_ze)

##################################################################################

def get_weight_from_weightmap(model,datasets,\
                   mc_datasets=None,\
                   mc_p_en=None,\
                   mc_p_ty=None,\
                   mc_p_ze=None,\
                   mc_p_we=1.,\
                   mc_p_ts=1.,\
                   mc_p_gw=1.,\
                   **model_kwargs):
    """
    Get weights for weighted datasets (generation spectra is already the target flux)
  
    Args:
        model (func): Not used, only for compatibility
        datasets (dict): used to provide nfiles
        
    Keyword Args:
        mc_p_en (array-like): primary energy
        mc_p_ty (array-like): primary particle type
        mc_p_ze (array-like): primary particle cos(zenith)
        mc_p_we (array-like): weight for mc primary, e.g. some interaction probability
        mc_p_gw (array-like): generation weight
        mc_p_ts (array-like): mc timescale
        mc_datasets (array-like): an array which has per-event dataset information
    
    Returns (array-like): Weights
    """
    #timescale    = np.zeros(len(mc_p_ts))
    all_ts = 0
    factors = np.ones(len(mc_p_ts))
    ts = {ds : mc_p_ts[mc_datasets==ds][0] for ds in list(datasets.keys())}
    for ds in list(datasets.keys()):
        #ts = datasets[ds]*mc_p_ts[mc_datasets==ds][0]
        #timescale[mc_datasets==ds] += datasets[ds]*mc_p_ts[mc_datasets==ds]
        #mc_p_we[mc_datasets==ds]*=(datasets[ds]*mc_p_ts[mc_datasets==ds])
        #all_ts += ts
        factors[mc_datasets == ds] /= (ts[ds]*datasets[ds])
    all_ts = sum([ts[x]*datasets[x] for x in list(datasets.keys())])
    #print all_ts
    #print factors[0]
    #print mc_p_we[0]
    #print mc_p_gw[0]
    weight = (factors*np.array(mc_p_gw)*np.array(mc_p_we))#/all_ts
    #weight = mc_p_gw*mc_p_we/factors
    if len(datasets) == 1:
        weight /= all_ts
    #print weight.sum()  
    return weight
  
  
  
  
