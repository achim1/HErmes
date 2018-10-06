"""
Tools for managing iterables
"""
from __future__ import division

from builtins import range
from past.utils import old_div

import numpy as np
import array as arr

from . import logger
Logger = logger.Logger



def slicer(list_to_slice, slices):
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
        yield thisslice

#######################################################################

def flatten(iterable_of_iterables):
    """
    Concat an iterable of iterables in a single iterable, chaining its elements

    Args:
        iterable_of_iterables (iterable): Currently supported is dimensionality of 2

    Returns:
        np.ndarray
    """
    # array is factor2 faster and uses 30% less memory than list
    flattened = arr.array("d", [])
    for k in iterable_of_iterables:
        flattened.extend(k)
    return np.asarray(flattened)


######################################################################

def multiplex(iterable, iterable_of_iterables):
    """
    More or less the inverst to flatten. Adjust the shape of iterable
    to match that of iterable_of_iterables by stretching each value to 
    be an iterable with the length of the respective element
    in iterable_of_iterables, but with always the same value

    Args:
        iterable : The array to be multiplexed
        iterable_of_iterables : The shape to be matched

    Returns:
        np.ndarray
    """
    
    lengths = [len(k) for k in iterable_of_iterables]
    assert (len(iterable) == len(lengths), "Can not multiplex unequal sized iterables!")

    multiplexed = [data*np.ones(lengths[i]) for i, data in enumerate(iterable)]
    multiplexed = np.asarray(multiplexed)
    return multiplexed  
