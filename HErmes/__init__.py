"""
A package for filtering datasets as common in high energy physics. Read data from hdf or
root files, classify the data in different categories and provide and easy interface
to easy access to the variables stored in the files.

The HErmes modules provides the following submodules:

- `selection` : Start from a .json configuration file to create a full fledged dataset which acts as a container for in different categories.

- `utils` : Aggregator for files and logging.

- `fitting` : Fit models to variable distributions with iminuit.

- `visual` : Data visualization.

- `icecube_goodies` : Weighting for icecube datasets.

- `analysis` : convenient functions for data analysis and working with distributions.

"""

__version__ = '0.1.0'
__all__ = ["fitting", "icecube_goodies", "utils",\
           "selection", "visual", "analysis"]

import os.path
import hepbasestack as hep

from . import utils
loglevel = hep.logger.LOGLEVEL

from . import visual

def set_loglevel(level):
    """
    Set the loglevel, 10 = debug, 20 = info, 30 = warn
    """
    hep.logger.LOGLEVEL = level
    return

# FIXME documentation
def _hook():
    pass
