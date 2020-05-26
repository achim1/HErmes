"""
Provide an easy-to-use, intuitive way of fitting models with different components
to data. The focus is less on a statistical sophisticated fitting rather than on 
an explorative approach to data investigation. This might help answer questions of 
the form - "How compatible is this data with a Gaussian + Exponential?".
Out of the box, this module provides tools targeted to a least-square fit, however,
in principle this could be extended to likelihood fits.

Currently the generation of the minimized error function is automatic, and it is 
generated *only* for the least-squares case, however this might be expanded in the 
future.

"""

from .functions import poisson, gauss, exponential, calculate_chi_square, fwhm_gauss
from .model import Model
