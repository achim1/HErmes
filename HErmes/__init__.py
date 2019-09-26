"""
A package for filtering datasets as common in high energy physics. Read data from hdf or
root files, classify the data in different categories and provide and easy interface
to easy access to the variables stored in the files.

The HErmes modules provides the following submodules:

- `selection` : Start from a .json configuration file to create a full fledged dataset which acts as a container for in different categories.

- `utils` : Aggregator for files and logging.

- `fitting` : Fit models to variable distributions with iminuit.

- `plotting` : Data visualization.

- `icecube_goodies` : Weighting for icecube datasets.

- `analysis` : convenient functions for data analysis and working with distributions.

"""

__version__ = '0.0.9dev'
__all__ = ["fitting", "icecube_goodies", "utils",\
           "selection", "plotting", "analysis"]

import os.path

from . import utils
loglevel = utils.logger.LOGLEVEL

from . import plotting

def set_loglevel(level):
    """
    Set the loglevel, 10 = debug, 20 = info, 30 = warn
    """
    utils.logger.LOGLEVEL = level
    return

# FIXME documentation
def _hook():
    pass
