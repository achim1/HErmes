"""
A set of
"""

from .plotting import VariableDistributionPlot

import matplotlib.style.core as c
import os.path

def add_styles():
    # reload the style
    USER_STYLES = os.path.dirname(__file__)
    #USER_STYLES.append(os.path.join(os.path.dirname(__file__), 'HErmes-default.mplstyle'))
    c.USER_LIBRARY_PATHS.append(USER_STYLES)
    c.reload_library()
    return None


add_styles()

