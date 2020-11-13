"""
Goodies for icecube
"""

import tables
import numpy as np
import os

GEOMETRY=os.path.join(os.path.split(__file__)[0],"geometry_ic86.h5")

class IceCubeGeometry(object):
    """
    Provide icecube geometry information
    """

    def __init__(self):
        self.geofile = None
        self.geo = None
        self.load_geo()

    def load_geo(self):
        """
        Load geometry information
        """

        #FIXME absolute path
        self.geofile = tables.open_file(GEOMETRY)
        self.geo = self.geofile.root.geometry.read()

    def coordinates(self, string, dom):
        """
        Calculate the xy position of a given string
        """
        #string -= 1

        assert string in range(1, 86), "String {} must be a number from 1 - 86".format(string)
        assert dom in range(1, 61), "Dom {} must be a number from 1 - 60".format(dom)

        mask = (self.geo['om'] == dom) & (self.geo['string'] == string)
        position = self.geo[mask]
        pos      = np.array([np.float(position['x']),np.float(position['y']),np.float(position['z'])])
        return pos

    def __del__(self):
        self.geofile.close()




