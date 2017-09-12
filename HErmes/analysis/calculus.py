"""
Common calculations
"""

import numpy as np

def opening_angle(reco_zen, reco_azi, true_zen, true_azi):
    """
    Calculate the opening angle between two vectors, described 
    by azimuth and zenith in some coordinate system. 
    Can be useful for estimatiion of angular uncertainty 
    for some reconstruction.
    Zenith and Azimuth in radians.

    Args:
        reco_zen (float): zenith of vector A
        reco_azi (float): azimuth of vector A
        true_zen (float): zenith of vector B
        true_azi (float): azimuth of vector B

    Returns:
        float: Opening angle in degree
    """

    sinisy = lambda x,y : np.sin(x)*np.sin(y)
    cosisy = lambda x,y : np.cos(x)*np.cos(y)

    c = np.cos
    
    # dot product
    cospsi = ((c(reco_zen)*c(true_zen)) + (sinisy(reco_zen,true_zen)*cosisy(reco_azi,true_azi)) + (sinisy(reco_zen,true_zen)*sinisy(reco_azi,true_azi)))
    psi = np.arccos(cospsi)
    psi = np.rad2deg(psi)
    return psi




