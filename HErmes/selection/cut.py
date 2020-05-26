"""
Remove part of the data which falls below a certain criteria.
"""

from .variables import Variable as V

from collections import defaultdict
from copy import deepcopy as copy
import operator
operator_lookup = {\
    ">"  : operator.gt,\
    "==" : operator.eq,\
    "!=" : operator.ne,\
    "<"  : operator.lt,\
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
            type (str)      : either 'AND' (default) or 'OR'

        Returns:
            HErmes.selection.Cut
        """

        self.condition = None
        self.name = None
        self.type = None

        if "condition" in kwargs:
            self.condition = kwargs["condition"]
        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'type' in kwargs:
            self.type = kwargs['type']
        else:
            self.type = 'AND'

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
        newcut = copy(self)
        assert self.type == other.type, "At the moment, mixing cut types is not supported"
        for k in other.cutdict:
            if k in self.cutdict:
                newcut.cutdict[k] += other.cutdict[k]
            else:
                newcut.cutdict[k] = other.cutdict[k]
        if other.condition is None:
            pass
        else:
            # condition is dict catname -> np.ndarray(bool)
            if newcut.condition is None:
                newcut.condition = other.condition
            else:
                for k in other.condition:
                    if k in self.condition:
                        newcut.condition[k] = np.logical_and(self.condition[k],\
                                                           other.condition[k])
                    else:
                        newcut.condition[k] = other.condition[k]
        return newcut

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
            rep = """< {} Cut with condition | \n""".format(self.type)
        else:
            rep = """< {} Cut | \n""".format(self.type)
        for i, (j, k) in sorted(self):
            rep += """| {0} {1} {2} \n""".format(i,inv_operator_lookup[j],k)

        rep += """| >"""
        return rep




