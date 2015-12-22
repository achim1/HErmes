"""
Define all variables here
"""

import numpy as n


from variables.variables import Variable as V

energy = V("energy",bins=n.linspace(0,10,20),transform=n.log10,label=r"$\log(E_{rec}/$GeV$)$",definitions=[("CredoFit","energy")])

