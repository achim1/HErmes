"""
Flux models for atmospheric neutrino and muon fluxes
as well as power law fluxes
"""

from icecube import icetray,dataclasses,NewNuFlux
import icecube.weighting.fluxes as mufluxes
import conversions as conv

import numpy as np

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

    # closure:)
    def flux(mc_p_energy,mc_p_type,mc_p_zenith):
        # weighting API requires second and third argument even if we
        # don't need it here
        flux = fluxconst * np.power(mc_p_energy, gamma)
        return flux
    return flux

###############################################


def AtmosphericNuFlux(modelname='honda2006',knee=False,fluxconst=1.):
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
            return fluxconst*nuflux.getFlux(mc_p_type,mc_p_energy,mc_p_zenith)
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


class NuFluxes:
    """
    Namespace for neutrino fluxes
    """
    Honda2006    = staticmethod(AtmoWrap("honda2006"))
    Honda2006H3a = staticmethod(AtmoWrap("honda2006",knee="gaisserH3a_elbert")) 
    Honda2006H4a = staticmethod(AtmoWrap("honda2006",knee="gaisserH4a_elbert"))
    ERS          = staticmethod(AtmoWrap("sarcevic_std"))
    ERSH3a       = staticmethod(AtmoWrap("sarcevic_std",knee="gaisserH3a_elbert"))
    ERSH4a       = staticmethod(AtmoWrap("sarcevic_std",knee="gaisserH4a_elbert"))
    BERSSH3a     = staticmethod(AtmoWrap("BERSS_H3a_central"))
    BERSSH4a     = staticmethod(AtmoWrap("BERSS_H3p_central"))
    E2           = staticmethod(PowerWrap())
    BARTOL       = staticmethod(AtmoWrap("bartol"))


class MuFluxes:
    """
    Namespace for atmospheric muon fluxes
    """
    GaisserH3a   = staticmethod(mufluxes.GaisserH3a)
    GaisserH4a   = staticmethod(mufluxes.GaisserH4a)
    Hoerandel    = staticmethod(mufluxes.Hoerandel)
    Hoerandel5   = staticmethod(mufluxes.Hoerandel5)
    HoerandelIT  = staticmethod(mufluxes.Hoerandel_IT)
    TIG1996      = staticmethod(mufluxes.TIG1996)
    GlobalFitGST = staticmethod(mufluxes.GlobalFitGST)
    Honda2004    = staticmethod(mufluxes.Honda2004)
