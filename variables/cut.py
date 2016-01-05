"""
Holds a class which allows to perform cuts on 
categories
"""

from pyevsel.variables.variables import Variable as V

import pandas as pd
import numpy as n

def _lambdanator(operation,value):
    return compile("lambda x : x %s" + str(value))


class Cut(object):
    """
    Impose criteria on variables of a certain
    category to reduce its mightiness
    """
    _cutdict = dict()
    _condition_map = {">" : "gt","<" : "lt",">=" : "ge","<=" : "le"}

    def __init__(self,variables=[("mc_p_energy",">=",5)]):

        for var,operation in variables:
            if isinstance(var,V):
                name = var.name
            if isinstance(var,str):
                name = var
            self._cutdict[name] = operation
        self.maskdict = dict()


    #def __call__(self,category):
    #    """
    #    do it!
    #    """
    #    newcat = category
    #    total_mask = n.ones(len(category),dtype=n.bool)
    #    for v in self._cutdict.keys():
    #        ops = self._cutdict[v]
    #        if not callable(ops):
    #            ops = self._condition_map[ops]
    #            newcat.vardict[v].data = category.vardict[v].data.__getattribute__(ops)()
    #        else:
    #            mask = category.vardict[v].data.map(ops)
    #            self.maskdict[v] = mask
    #            total_mask = n.logical_and(total_mask,mask)
    #    print len(total_mask) == len(total_mask[n.isfinite(total_mask)])
    #    print total_mask
    #    total_mask = n.array(total_mask)
    #    for v in category.vardict.keys():
    #        print v
    #        print category.vardict[v].data
    #        newcat.vardict[v].data = category.vardict[v].data.where(total_mask)
    #    return newcat

class CutCollection:
    pass

