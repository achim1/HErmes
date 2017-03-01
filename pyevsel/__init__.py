"""
HEP style eventselection with python
"""
from __future__ import print_function

import atexit
import os
import appdirs
import shutil

from matplotlib import get_configdir as mpl_configdir

__version__ = '0.0.6'
__all__ = ["fitting", "icecube_goodies", "utils", "variables", "plotting"]


_appdir = os.path.split(__file__)[0]
_appname = os.path.split(_appdir)[1]
# the config files
STYLE_BASEFILE_STD = os.path.join(_appdir,os.path.join("plotting","pyevselpresent.mplstyle"))
STYLE_BASEFILE_PRS = os.path.join(_appdir,os.path.join("plotting","pyevseldefault.mplstyle"))
PLOTSCFG_BASEFILE = os.path.join(_appdir,os.path.join("plotting","plotsconfig.yaml"))
PATTERNFILE = os.path.join(_appdir,os.path.join("utils","PATTERNS.cfg"))

def get_configdir():
    """
    Definges a configdir for this package under $HOME/.pyevsel
    """
    config_dir = appdirs.user_config_dir(_appname)
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    return config_dir

def install_config(style_default=STYLE_BASEFILE_STD, \
                   style_present=STYLE_BASEFILE_PRS, \
                   plots_config=PLOTSCFG_BASEFILE, \
                   patternfile=PATTERNFILE):
    """
    Sets up configuration and style files

    Keyword Args:
        style_default (str): location of style file to use by defautl
        style_present (str): location of style file used for presentations
        plots_config (str): configureation file for plots
        patternfile (str): location of patternfile with file patterns to search and read
    """

    print ("Installing config...")
    cfgdir = get_configdir()
    mpl_styledir = os.path.join(mpl_configdir(),"stylelib")
    for f in style_default, style_present:
        assert os.path.exists(f), "STYLEFILE {} missing... indicates a problem with some paths or corrupt packege. Check source code location".format(f)
        shutil.copy(f,mpl_styledir)

    for f in plots_config, patternfile:
        assert os.path.exists(f), "STYLEFILE {} missing... indicates a problem with some paths or corrupt packege. Check source code location".format(f)
        shutil.copy(f,cfgdir)

def _DeleteTmpFile():
    """
    Remove the created tmp files
    when interpreter is ended
    """

    from pyevsel.plotting import CONFIGFILE
    os.remove(CONFIGFILE)

install_config()

try:
    atexit.register(_DeleteTmpFile)
except IOError:
    print("Can not register tmpfile deletion right now...")

