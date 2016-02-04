"""
HEP style eventselection with python
"""


import atexit
import os

__version__ = '0.0.1'
# delete created tmpfile
# a bit ugly though..as

def _DeleteTmpFile():
    """
    Remove the created tmp files
    when interpreter is ended
    """

    from pyevsel.plotting import CONFIGFILE
    os.remove(CONFIGFILE)

try:
    atexit.register(_DeleteTmpFile)
except IOError:
    print "Can not register tmpfile deletion right now..."

