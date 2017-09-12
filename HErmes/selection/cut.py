"""
Remove part of the data which falls below a certain criteria.
"""

from __future__ import absolute_import


from builtins import object
from .variables import Variable as V

from collections import defaultdict
from copy import deepcopy as copy
import operator
operator_lookup = {\
    ">" : operator.gt,\
    "==" : operator.eq,\
    "<" : operator.lt,\
    ">=" : operator.ge,\
    "<=" : operator.le\
    }

inv_operator_lookup = {v: k for k, v in list(operator_lookup.items())}


class Cut(object):
    """
    Cuts are basically conditions on a set of parameters.
    """

    def __init__(self, *cuts, **kwargs):
        """
        Args:
            cuts (list): like this [("mc_p_energy",">=",5)]

        Keyword Args:
            condition (dict): where to apply the cut. It has te be a dictionary of
                              categoryname to np.ndarray(bool)

        Returns:
            HErmes.selection.Cut
        """

        self.condition = None
        self.name = None
        if "condition" in kwargs:
            self.condition = kwargs["condition"]
        if 'name' in kwargs:
            self.name = kwargs['name']
        self.cutdict = defaultdict(list)
        for var, operation, value in cuts:
            # FIXME: most likely this has to go away...
            if isinstance(var, V):
                name = var.name
            if isinstance(var, str):
                name = var
            else:
                raise TypeError("Unable to understand variable type {}!".format(name))
            self.cutdict[name].append((operator_lookup[operation],value))

    @property
    def variablenames(self):
        """
        The names of the variables the cut will be applied to
        """
        return list(self.cutdict.keys())

    def __add__(self, other):
        new = copy(self)
        for k in other.cutdict:
            if k in self.cutdict:
                new.cutdict[k] += other.cutdict[k]
            else:
                new.cutdict[k] = other.cutdict[k]
        if other.condition is None:
            pass
        else:
            # condition is dict catname -> np.ndarray(bool)
            if new.condition is None:
                new.condition = other.condition
            else:
                for k in other.condition:
                    if k in self.condition:
                        new.condition[k] = np.logical_and(self.condition[k],\
                                                           other.condition[k])
                    else:
                        new.condition[k] = other.condition[k]
        return new

    def __iter__(self):
        """
        Return name, cutfunc pairs

        Yields:
            tuple
        """
        # flatten out the cutdict
        for k in list(self.cutdict.keys()):
            for j in self.cutdict[k]:
                yield k, j

    def __repr__(self):
        if self.condition is not None:
            rep = """< Cut with condition | \n"""
        else:
            rep = """< Cut | \n"""
        for i, (j, k) in sorted(self):
            rep += """| {0} {1} {2} \n""".format(i,inv_operator_lookup[j],k)

        rep += """| >"""
        return rep




