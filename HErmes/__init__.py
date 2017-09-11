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

__version__ = '0.0.7'
__all__ = ["fitting", "icecube_goodies", "utils",\
           "selection", "plotting", "analysis"]

from . import utils
loglevel = utils.logger.LOGLEVEL

