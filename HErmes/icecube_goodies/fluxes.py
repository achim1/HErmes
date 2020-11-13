"""
Flux models for atmospheric neutrino and muon fluxes
as well as power law fluxes
"""
from __future__ import absolute_import

import numpy as np
from functools import reduce
from . import conversions as conv

icecube_found = False
try:
    from icecube import icetray, dataclasses, NewNuFlux
    from icecube.weighting.weighting import from_simprod
    import icecube.weighting.fluxes as ICMuFluxes
    icecube_found = True
except ImportError:
    print("WARNING: module icecube not found!")

    # hobo mufluxes
    class ICMuFluxes(object):
        GaisserH3a = None
        GaisserH4a = None
        Hoerandel  = None
        Hoerandel5 = None

    def AtmosphericNuFlux(*args, **kwargs):
        hobo =  lambda x: x
        return hobo

AREA_SUM = 18946832.9035663

###############################################


def PowerLawFlux(fluxconst=1e-8,gamma=2):
    """
    A simple powerlaw flux

    Args:
        fluxconst (float): normalization
        gamma (float): spectral index

    Returns (func): the flux function

    """

    if gamma > 0:
        gamma *= -1

    def flux(mc_p_energy,mc_p_type,mc_p_zenith, fluxconst=fluxconst, gamma=gamma):
        # weighting API requires second and third argument even if we
        # don't need it here
        return fluxconst * np.power(mc_p_energy, gamma)

    return flux

###############################################

if icecube_found:

    def AtmosphericNuFlux(modelname='honda2006',knee=False, fluxconst=1.):
        """
        Create an Atmospheric neutrino flux

        Args:
            modelname (str): a atmospheric model
            knee (bool): ad a knee model
            fluxconst (float): scale flux by this const

        Returns (func): the requested flux. Takes energy, type and maybe cos(zenith) as parameters

        """
        nuflux = NewNuFlux.makeFlux(modelname)
        if knee:
            nuflux.knee_reweighting_model = knee

        def flux(mc_p_energy,mc_p_type,mc_p_zenith):
            mc_p_type = np.int32(mc_p_type)
            mc_p_type = conv.ConvertPrimaryFromPDG(mc_p_type)
            try:
                return fluxconst*nuflux.getFlux(mc_p_type,mc_p_energy,mc_p_zenith)
            except RuntimeError:
                if conv.IsPDGEncoded(mc_p_type,neutrino=True):
                    mc_p_type = conv.ConvertPrimaryFromPDG(mc_p_type)
                else:
                    mc_p_type = conv.ConvertPrimaryToPDG(mc_p_type)
                # FIXME: is this still the case?
                # the fluxes given by newnuflux are only for anti/neutrinos
                # so to calculate the total flux, there is a number of 2 necessary
                return 2*fluxconst*nuflux.getFlux(mc_p_type,mc_p_energy,mc_p_zenith)
        return flux

##############################################


def AtmoWrap(*args,**kwargs):
    """
    Allows currying atmospheric flux functions
    for class interface
    Args:
        *args: passed through to AtmosphericNuFlux
        **kwargs: passed through to AtmosphericNuFlux

    Returns: AtmosphericNuFlux with applied arguments

    """
    def wrapper():
        return AtmosphericNuFlux(*args,**kwargs)
    return wrapper

##############################################


def PowerWrap(*args,**kwargs):
    """
    Allows currying PowerLawFlux for class interface

    Args:
        *args: applied to PowerLawFlux
        **kwargs: applied to PowerLawFlux

    Returns: PowerLawFlux with applied arguments

    """

    def wrapper():
        return PowerLawFlux(*args,**kwargs)
    return wrapper

##############################################

def generated_corsika_flux(ebinc,datasets):
    """
    Calculate the livetime of a number of given coriska datasets using the weighting moduel
    The calculation here means a comparison of the number of produced events per energy bin
    with the expected event yield from fluxes in nature. If necessary call home to the simprod db.
    Works for 5C datasets.

    Args:
        ebinc (np.array): Energy bins (centers)
        datasets (list): A list of dictionaries with properties of the datasets or dataset numbers. If only nu8mbers are given, then simprod db will be queried
            format of dataset dict:
            example_datasets ={42: {"nevents": 1,\
                   "nfiles": 1,\
                   "emin": 1,\
                   "emax": 1,\
                   "normalization": [10., 5., 3., 2., 1.],\
                   "gamma": [-2.]*5,\
                   "LowerCutoffType": 'EnergyPerNucleon',\
                   "UpperCutoffType": 'EnergyPerParticle',\
                   "height": 1600,\
                   "radius": 800}}

    Returns:
        tuple (generated protons, generated irons)
    """
    
    if isinstance(datasets,dict):
        pass

    elif not isinstance(datasets,list):
        datasets = list(datasets)

    generators = []
    for ds in datasets:
       
        if not isinstance(datasets,dict):
            assert len(list(ds.values())) == 1, "Too many arguments per dataset"

        if isinstance(ds,int):
            db_result = from_simprod(ds)
            if isinstance(db_result, tuple):
                db_result = db_result[1]
            generators.append(db_result*datasets[ds])

        elif isinstance(list(ds.values())[0],int):
            db_result = from_simprod(int(list(ds.keys())[0]))
            if isinstance(db_result,tuple):
                db_result = db_result[1]
            generators.append(db_result*list(ds.values())[0])
        elif isinstance(list(ds.values())[0],dict):
            nfiles = ds.pop("nfiles")
            generators.append(nfiles*FiveComponent(**ds))
        else:
            raise ValueError("Problems understanding dataset properties {}".format(ds.__repr__()))

    gensum = reduce(lambda x, y: x + y, generators)
    p_gen  = AREA_SUM*gensum(ebinc,2212)
    fe_gen = AREA_SUM*gensum(ebinc,1000260560)

    return p_gen, fe_gen

##################################################################


class NuFluxes(object):
    """
    Namespace for neutrino fluxes
    """
    Honda2006    = staticmethod(AtmosphericNuFlux("honda2006"))
    Honda2006H3a = staticmethod(AtmosphericNuFlux("honda2006",knee="gaisserH3a_elbert")) 
    Honda2006H4a = staticmethod(AtmosphericNuFlux("honda2006",knee="gaisserH4a_elbert"))
    ERS          = staticmethod(AtmosphericNuFlux("sarcevic_std"))
    ERSH3a       = staticmethod(AtmosphericNuFlux("sarcevic_std",knee="gaisserH3a_elbert"))
    ERSH4a       = staticmethod(AtmosphericNuFlux("sarcevic_std",knee="gaisserH4a_elbert"))
    BERSSH3a     = staticmethod(AtmosphericNuFlux("BERSS_H3a_central"))
    BERSSH4a     = staticmethod(AtmosphericNuFlux("BERSS_H3p_central"))
    E2           = staticmethod(PowerLawFlux())
    BARTOL       = staticmethod(AtmosphericNuFlux("bartol"))


class MuFluxes(object):
    """
    Namespace for atmospheric muon fluxes
    """
    GaisserH3a   = staticmethod(ICMuFluxes.GaisserH3a)
    GaisserH4a   = staticmethod(ICMuFluxes.GaisserH4a)
    Hoerandel    = staticmethod(ICMuFluxes.Hoerandel)
    Hoerandel5   = staticmethod(ICMuFluxes.Hoerandel5)

    # if these fluxes are available depends on the version
    # of the weighting module
    
    #HoerandelIT  = staticmethod(mufluxes.Hoerandel_IT)
    #TIG1996      = staticmethod(mufluxes.TIG1996)
    #GlobalFitGST = staticmethod(mufluxes.GlobalFitGST)
    #Honda2004    = staticmethod(mufluxes.Honda2004)
