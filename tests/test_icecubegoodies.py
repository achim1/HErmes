import pytest
import numpy as np

import HErmes.icecube_goodies.helpers as helpers

def test_geometry():
    geo = helpers.IceCubeGeometry()
    for string in range(1,86):
        print (string)
        for dom in range(1,61):
            c =  geo.coordinates(string, dom)
            assert len(c) == 3
            for k in c:
                assert np.isfinite(k)



