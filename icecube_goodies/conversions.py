"""
Unit conversions and such
"""

from numpy import vectorize

####################################################

class ParticleType:
    PPlus       =   14
    He4Nucleus  =  402
    N14Nucleus  = 1407
    O16Nucleus  = 1608
    Al27Nucleus = 2713
    Fe56Nucleus = 5626
    NuE         =   66
    NuEBar      =   67
    NuMu        =   68
    NuMuBar     =   69
    NuTau       =  133
    NuTauBar    =  134

#######################################################

class PDGCode:
    PPlus       =       2212
    He4Nucleus  = 1000020040
    N14Nucleus  = 1000070140
    O16Nucleus  = 1000080160
    Al27Nucleus = 1000130270
    Fe56Nucleus = 1000260560
    NuE         =         12
    NuEBar      =        -12
    NuMu        =         14
    NuMuBar     =        -14
    NuTau       =         16
    NuTauBar    =        -16

#########################################################

ptype_to_pdg = \
{        14    :        2212,
        402    :  1000020040,
       1407    :  1000070140,
       1608    :  1000080160,
       2713    :  1000130270,
       5626    :  1000260560,
         66    :          12,
         67    :         -12,
         68    :          14,
         69    :         -14,
        133    :          16,
        134    :         -16}

###########################################################

pdg_to_ptype = \
{       2212   :    14,
1000020040     :   402,
1000070140     :  1407,
1000080160     :  1608,
1000130270     :  2713,
1000260560     :  5626,
          12   :    66,
       -12     :    67,
        14     :    68,
       -14     :    69,
        16     :   133,
       -16     :   134}

##############################

def ConvertPrimaryToPDG(pid):
    """
    Convert a primary id in an i3 file to the new values
    given by the pdg
    """
    def _convert(pid):
        if ptype_to_pdg.has_key(pid):
            return ptype_to_pdg[pid]
        else:
             return pid

    return vectorize(_convert)(pid)

###############################

