"""
Miscellaneous tools
"""

from __future__ import division

__all__ = ["files"]


from hepbasestack import logger
from hepbasestack.itools import slicer, flatten, multiplex
from hepbasestack import timeit, isnotebook

from . import files

Logger = logger.Logger

