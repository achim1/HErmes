import pytest
import numpy as np
from HErmes.analysis import calculus, fluxes, tasks

def test_opening_angle():

    z1 = np.zeros(5)
    a1 = np.zeros(5)

    z2 = (np.pi/2)*np.ones(5)
    a2 = np.zeros(5)

    z3 = -(np.pi/2)*np.ones(5)
    

    #print calculus.opening_angle(0,0,np.pi/2,0)     
    #print calculus.opening_angle(z1,a1,z2,a2)
    assert (calculus.opening_angle(z1, a1, z1, a1)).sum() == 0
    assert (calculus.opening_angle(z1, a1, z2, a2)).sum() == 5*90.
    assert (calculus.opening_angle(z2, a2, z3, a2)).sum() == 5*180


def test_powerlaw_flux():

    pl = fluxes.PowerLawFlux(10,100,1,-1)
    assert isinstance(pl.fluxsum(), float)
    assert pl.fluxsum() == pl.phi0 * (np.log(100) - np.log(10))
    pl = fluxes.PowerLawFlux(10,100,1,-2)
    assert pl.E2_1E8(42.) == 1e-8*np.power(42.,-2)
    assert isinstance(pl.fluxsum(), float)
    assert isinstance(pl(np.array([1,2,3,4])), np.ndarray)
    pl = fluxes.PowerLawFlux(10,100,1,42)
    with pytest.raises(ValueError):
        pl.fluxsum()


def test_constant():
    c = fluxes.Constant()
    assert c.identity([1,2,3,4,5]).sum() == 5

def test_construct_slices():
    labels, cuts = tasks.construct_slices("test", np.linspace(0,100,10))
    assert len(labels) == len(cuts) == 9
    
    




