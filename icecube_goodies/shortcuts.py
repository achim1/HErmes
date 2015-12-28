"""
Convenient shortcuts
"""
from icecube.weighting.fluxes import *
from weighting import AtmosphericNuFlux,PowerLawFlux

def AtmoWrap(*args,**kwargs):
    def wrapper():
        return AtmosphericNuFlux(*args,**kwargs)
    return wrapper

def PowerWrap(*args,**kwargs):
    def wrapper():
        return PowerLawFlux(*args,**kwargs)
    return wrapper

class NuFluxes:
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
    GaisserH3a   = staticmethod(GaisserH3a)
    GaisserH4a   = staticmethod(GaisserH4a)
    Hoerandel    = staticmethod(Hoerandel)
    Hoerandel5   = staticmethod(Hoerandel5)
    HoerandelIT  = staticmethod(Hoerandel_IT)
    TIG1996      = staticmethod(TIG1996)
    GlobalFitGST = staticmethod(GlobalFitGST)
    Honda2004    = staticmethod(Honda2004)
