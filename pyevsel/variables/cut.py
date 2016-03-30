"""
Use with pyevsel.categories.Category to perfom cuts
"""

from pyevsel.variables.variables import Variable as V

import operator
operator_lookup = {\
    ">" : operator.gt,\
    "==" : operator.eq,\
    "<" : operator.lt,\
    ">=" : operator.ge,\
    "<=" : operator.le\
    }

def create_func(operation,value):
    """operator_lookup
    Create conditional function for the provide operator
    value pair

    Args:
        operation (str): an operator string, '<', '==", etc
        value (float): goes with operation

    Returns:
        func
    """
    def func(x):
        return operator_lookup[operation](x,value)
    return func

class Cut(object):
    """
    Impose criteria on variables of a certain
    category to reduce its mightiness
    """

    def __init__(self,*cuts,**kwargs):
        """
        Create a new cut, with the variables and operations
        given in cuts

        Args:
            cuts (list): like this [("mc_p_energy",">=",5)]

        Keyword Args:
            condition (numpy.ndarry(bool)): where to apply the cut

        Returns:
            None
        """

        self.condition = None
        if "condition" in kwargs:
            self.condition = kwargs["condition"]
        self.cutdict = dict()
        self.compiled_cuts = dict()
        for var,operation,value in cuts:
            if isinstance(var,V):
                name = var.name
            if isinstance(var,str):
                name = var
            self.cutdict[name] = (operator_lookup[operation],value)

            # The idea of compiled cuts
            # and using pandas.Series.apply is
            # nice, but too slow!
            #self.compiled_cuts[name] = create_func(operation,value)

    def __iter__(self):
        """
        Return name, cutfunc pairs
        """

        for k in self.cutdict.keys():
            #yield k,self.compiled_cuts[k]
            yield k,self.cutdict[k]

    def __repr__(self):
        rep = """<Cut """
        for k in self.cutdict.keys():
            rep += """|%s %s %4.2f """ %(k,self.cutdict[k][0],self.cutdict[k][1])

        rep += """>"""
        return rep

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
