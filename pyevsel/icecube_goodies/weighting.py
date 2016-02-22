"""
An interface to icecube's weighting schmagoigl
"""

from icecube.weighting.weighting import from_simprod, EnergyWeight,ParticleType
import inspect
import conversions as conv

reload(conv)

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
        #print zenith
        if mapping:
            pmap = {14:ParticleType.PPlus, 402:ParticleType.He4Nucleus, 1407:ParticleType.N14Nucleus, 2713:ParticleType.Al27Nucleus, 5626:ParticleType.Fe56Nucleus}
            ptype = map(lambda x : pmap[x], ptype )

        # FIXME: This is too ugly and not general
        can_use_zenith = False
        if hasattr(self.flux,"__call__"):
            if hasattr(self.flux.__call__,"im_func"):
                args = inspect.getargs(self.flux.__call__.im_func.func_code)
                if len(args.args) == 4: # account for self
                    can_use_zenith = True
            else:
                can_use_zenith = True # method wrapper created by NewNuflux 
        else:
            args = inspect.getargs(self.flux.func_code) 
            if len(args.args) == 3:
                can_use_zenith = True       

        if (len(zenith) > 0) and can_use_zenith:
            return self.flux(energy,ptype,zenith)/self.gen(energy,particle_type=ptype,cos_theta=zenith)
        else:
            return self.flux(energy,ptype)/self.gen(energy,particle_type=ptype,cos_theta=zenith)

###########################################


def GetGenerator(datasets):
    """
    datasets must be a dict of dataset_id : number_of_files

    Args:
        datasets (dict): Query the database for these datasets.
                         dict dataset_id -> number of files
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

    Args:
        dataset (int): dataset_id
    """
    nfiles,__ = from_simprod(dataset)
    return nfiles

##########d##################################


def GetModelWeight(model,datasets,mc_p_energy=[],mc_p_type=[],mc_p_zenith=[],**model_kwargs):
    """
    Compute weights using a predefined model

    Args:
        model (func): Used to calculate the target flux
        datasets (dict): Get the generation pdf for these datasets from the db
                         dict needs to be dataset_id -> nfiles
    Keyword Args:
        mc_p_energy (array-like): primary energy
        mc_p_type (array-like): primary particle type
        mc_p_zenith (array-like): primary particle cos(zenith)
    """
    if model_kwargs:
        flux = model(**model_kwargs)
    else:
        flux = model()

    gen  = GetGenerator(datasets)
    weight = Weight(gen,flux)
    return weight(mc_p_energy,mc_p_type,zenith=mc_p_zenith)

