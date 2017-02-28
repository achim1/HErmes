"""
Tools for managing iterables
"""
from __future__ import division

from builtins import range
from past.utils import old_div
from . import logger
Logger = logger.Logger

def slicer(list_to_slice,slices):
    """
    *Generator* Slice a list in  individual slices

    Args:
        list_to_slice (list): a list of items
        slices (int): how many lists will be sliced of
    
    Returns:
        list: slices of the given list
    """

    #FIXME: there must be something in the std lib..
    # implementing this because I am flying anyway
    # right now and have nothing to do..

    if slices == 0:
        slices = 1 # prevent ZeroDivisionError
    maxslice = old_div(len(list_to_slice),slices)
    if (maxslice*slices) < len(list_to_slice) :
        maxslice += 1
    Logger.info("Sliced list in {} slices with a maximum slice index of {}" .format(slices,maxslice))
    for index in range(0,slices):
        lower_bound = index*maxslice
        upper_bound = lower_bound + maxslice
        thisslice = list_to_slice[lower_bound:upper_bound]
        #if not thisslice: #Do not emit empty lists!
        yield thisslice


