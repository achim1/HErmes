"""
Unit conversions and such
"""

from numpy import vectorize,int32
#from icecube import icetray,dataclasses,NewNuFlux
####################################################


#FIXME: boilerplate this here, because it is not possible
# to load two versions of icerec simultaniously
# This should go into icerec!

class ParticleType(object):
    """
    Namespace for icecube particle type codes
    """
    unknown = 0
    Gamma = 1
    EPlus = 2
    EMinus = 3
    MuPlus = 5
    MuMinus = 6
    Pi0 = 7
    PiPlus = 8
    PiMinus = 9
    K0_Long = 10
    KPlus = 11
    KMinus = 12
    Neutron = 13
    PPlus = 14
    PMinus = 15
    K0_Short = 16
    NuE = 66
    NuEBar = 67
    NuMu = 68
    NuMuBar = 69
    TauPlus = 131
    TauMinus = 132
    NuTau = 133
    NuTauBar = 134
    He4Nucleus = 402
    Li7Nucleus = 703
    Be9Nucleus = 904
    B11Nucleus = 1105
    C12Nucleus = 1206
    N14Nucleus = 1407
    O16Nucleus = 1608
    F19Nucleus = 1909
    Ne20Nucleus = 2010
    Na23Nucleus = 2311
    Mg24Nucleus = 2412
    Al26Nucleus = 2613
    Al27Nucleus = 2713
    Si28Nucleus = 2814
    P31Nucleus = 3115
    S32Nucleus = 3216
    Cl35Nucleus = 3517
    Ar36Nucleus = 3618
    Ar37Nucleus = 3718
    Ar38Nucleus = 3818
    Ar39Nucleus = 3918
    Ar40Nucleus = 4018
    Ar41Nucleus = 4118
    Ar42Nucleus = 4118
    K39Nucleus = 3919
    Ca40Nucleus = 4020
    Sc45Nucleus = 4521
    Ti48Nucleus = 4822
    V51Nucleus = 5123
    Cr52Nucleus = 5224
    Mn55Nucleus = 5525
    Fe56Nucleus = 5626

#######################################################

class PDGCode(object):
    """
    Namespace for PDG conform particle type codes
    """
    unknown = 0
    Gamma = 22
    EPlus = -11
    EMinus = 11
    MuPlus = -13
    MuMinus = 13
    Pi0 = 111
    PiPlus = 211
    PiMinus = -211
    K0_Long = 130
    KPlus = 321
    KMinus = -321
    Neutron = 2112
    PPlus = 2212
    PMinus = -2212
    K0_Short = 310
    Eta = 221
    Lambda = 3122
    SigmaPlus = 3222
    Sigma0 = 3212
    SigmaMinus = 3112
    Xi0 = 3322
    XiMinus = 3312
    OmegaMinus = 3334
    NeutronBar = -2112
    LambdaBar = -3122
    SigmaMinusBar = -3222
    Sigma0Bar = -3212
    SigmaPlusBar = -3112
    Xi0Bar = -3322
    XiPlusBar = -3312
    OmegaPlusBar = -3334
    DPlus = 411
    DMinus = -411
    D0 = 421
    D0Bar = -421
    DsPlus = 431
    DsMinusBar = -431
    LambdacPlus = 4122
    WPlus = 24
    WMinus = -24
    Z0 = 23
    NuE = 12
    NuEBar = -12
    NuMu = 14
    NuMuBar = -14
    TauPlus = -15
    TauMinus = 15
    NuTau = 16
    NuTauBar = -16
    He3Nucleus = 1000020030
    He4Nucleus = 1000020040
    Li6Nucleus = 1000030060
    Li7Nucleus = 1000030070
    Be9Nucleus = 1000040090
    B10Nucleus = 1000050100
    B11Nucleus = 1000050110
    C12Nucleus = 1000060120
    C13Nucleus = 1000060130
    N14Nucleus = 1000070140
    N15Nucleus = 1000070150
    O16Nucleus = 1000080160
    O17Nucleus = 1000080170
    O18Nucleus = 1000080180
    F19Nucleus = 1000090190
    Ne20Nucleus = 1000100200
    Ne21Nucleus = 1000100210
    Ne22Nucleus = 1000100220
    Na23Nucleus = 1000110230
    Mg24Nucleus = 1000120240
    Mg25Nucleus = 1000120250
    Mg26Nucleus = 1000120260
    Al26Nucleus = 1000130260
    Al27Nucleus = 1000130270
    Si28Nucleus = 1000140280
    Si29Nucleus = 1000140290
    Si30Nucleus = 1000140300
    Si31Nucleus = 1000140310
    Si32Nucleus = 1000140320
    P31Nucleus = 1000150310
    P32Nucleus = 1000150320
    P33Nucleus = 1000150330
    S32Nucleus = 1000160320
    S33Nucleus = 1000160330
    S34Nucleus = 1000160340
    S35Nucleus = 1000160350
    S36Nucleus = 1000160360
    Cl35Nucleus = 1000170350
    Cl36Nucleus = 1000170360
    Cl37Nucleus = 1000170370
    Ar36Nucleus = 1000180360
    Ar37Nucleus = 1000180370
    Ar38Nucleus = 1000180380
    Ar39Nucleus = 1000180390
    Ar40Nucleus = 1000180400
    Ar41Nucleus = 1000180410
    Ar42Nucleus = 1000180420
    K39Nucleus = 1000190390
    K40Nucleus = 1000190400
    K41Nucleus = 1000190410
    Ca40Nucleus = 1000200400
    Ca41Nucleus = 1000200410
    Ca42Nucleus = 1000200420
    Ca43Nucleus = 1000200430
    Ca44Nucleus = 1000200440
    Ca45Nucleus = 1000200450
    Ca46Nucleus = 1000200460
    Ca47Nucleus = 1000200470
    Ca48Nucleus = 1000200480
    Sc44Nucleus = 1000210440
    Sc45Nucleus = 1000210450
    Sc46Nucleus = 1000210460
    Sc47Nucleus = 1000210470
    Sc48Nucleus = 1000210480
    Ti44Nucleus = 1000220440
    Ti45Nucleus = 1000220450
    Ti46Nucleus = 1000220460
    Ti47Nucleus = 1000220470
    Ti48Nucleus = 1000220480
    Ti49Nucleus = 1000220490
    Ti50Nucleus = 1000220500
    V48Nucleus = 1000230480
    V49Nucleus = 1000230490
    V50Nucleus = 1000230500
    V51Nucleus = 1000230510
    Cr50Nucleus = 1000240500
    Cr51Nucleus = 1000240510
    Cr52Nucleus = 1000240520
    Cr53Nucleus = 1000240530
    Cr54Nucleus = 1000240540
    Mn52Nucleus = 1000250520
    Mn53Nucleus = 1000250530
    Mn54Nucleus = 1000250540
    Mn55Nucleus = 1000250550
    Fe54Nucleus = 1000260540
    Fe55Nucleus = 1000260550
    Fe56Nucleus = 1000260560
    Fe57Nucleus = 1000260570
    Fe58Nucleus = 1000260580

#############################################################
ptype = ParticleType()
pdgtype = PDGCode()

# wrap these to dicts
#ptype_to_pdg = dict([(ptype.__getattribute__(x),pdgtype.__getattribute__(x))\
#                     for x in dir(ParticleType)\
#                     if not x.startswith('_') and x in dir(PDGCode)])
ptype_to_pdg = {ptype.__getattribute__(x) : pdgtype.__getattribute__(x)\
                     for x in dir(ParticleType)\
                     if not x.startswith('_') and x in dir(PDGCode)}
#pdg_to_ptype = dict([(pdgtype.__getattribute__(x),ptype.__getattribute__(x))\
#                     for x in dir(PDGCode)\
#                     if not x.startswith('_') and x in dir(ParticleType)])
pdg_to_ptype = {pdgtype.__getattribute__(x) : ptype.__getattribute__(x)\
                     for x in dir(PDGCode)\
                     if not x.startswith('_') and x in dir(ParticleType)}


def IsPDGEncoded(pid,neutrino=False):
    """
    Check if the particle has already a pdg compatible
    pid

    Args:
        id (int): Partilce Id

    Keyword Args:
        neutrino (bool): as nue is H in PDG, set true if you know already
                    that ihe particle might be a neutrino

    Returns (bool): True if PDG compatible
    """

    for i in pid.values:
        if i in list(pdg_to_ptype.keys()):
            if i == 14 and neutrino:
                return True
            elif i == 14 and not neutrino:
                return False

            else:
                return True

    return False

##############################


def ConvertPrimaryToPDG(pid):
    """
    Convert a primary id in an i3 file to the new values
    given by the pdg
    """
    def _convert(pid):
        if pid in ptype_to_pdg:
            return int32(ptype_to_pdg[pid])
        else:
            return int32(pid)

    return vectorize(_convert)(pid)

###############################


def ConvertPrimaryFromPDG(pid):
    """
    Convert a primary id in an i3 file to the new values
    given by the pdg
    """
    def _convert(pid):
        if pid in pdg_to_ptype:
            return int32(pdg_to_ptype[pid])
        else:
            return int32(pid)

    return vectorize(_convert)(pid)

