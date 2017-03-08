"""
Simple to use fitting tools

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


from future import standard_library
standard_library.install_aliases()
from .functions import poisson, gauss, exponential, calculate_chi_square
from .model import Model
