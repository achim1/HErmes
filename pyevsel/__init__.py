"""
HEP style eventselection with python
"""


import atexit
import os

# delete created tmpfile
# a bit ugly though..as

def _DeleteTmpFile():
    """
    Remove the created tmp files
    when interpreter is ended
    """

    from pyevsel.plotting import CONFIGFILE
    os.remove(CONFIGFILE)

atexit.register(_DeleteTmpFile)