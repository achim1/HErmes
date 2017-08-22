"""
Investigate variables

"""
from __future__ import absolute_import
from collections import defaultdict
from ..selection.cut import Cut

def construct_slices(name, bins):
    """
    Prepare a set of cuts for the variable with name"name" in the dataset

    Args:
        name (str): The name of the variable in the dataset
        bins (array): bincenters of the slices

    Returns:
        tuple (list of strings, list of cuttuples)
    """
    l_binedges = bins[:-1]
    r_binedges = bins[1:]
    cuts = [[(name, ">", i), (name, "<=", j)] for i, j in zip(l_binedges, r_binedges)]
    labels = ["{:4.2f}".format(i) for i in bins]
    return labels, cuts



