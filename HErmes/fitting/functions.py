"""
Provide some simple functions which can be used to create models
"""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy.stats as st

#import statsmodels.stats.gof as gof

from future import standard_library
standard_library.install_aliases()

from ..utils import Logger

def poisson(x, lmbda):
    """
    Poisson probability

    Args:
        x (int): measured number of occurences
        lmbda (int): expected number of occurences

    Returns:
        np.ndarray
    """
    x = np.asarray(x, dtype=np.int32)
    pois = st.poisson(lmbda)
    return pois.pmf(x)
    #return np.power(lmbda, k) * np.exp(-1 * lmbda) / factorial(k)

################################################

def pandel_factory(c_ice):
    """
    Create a pandel function with the defined parameters

    Args:
        c_ice (float): group velocity in ice in m/ns

    Returns:
        callable
    """
    tau = 450.
    el = 47.
    ella = 98.

    if c_ice > 1:
        Logger.warning("Wrong unit for c_ice... multiplying with 1e-9")
        c_ice *= 1e-9

    from math import gamma
    gamma = np.vectorize(gamma)
    a = lambda tau, ella: 1 / tau + c_ice / ella
    b = lambda r, el: r / el

    def pandel(x, distance):
        """

        Args:
            x : time where the pandel is evalueated (in ns)
        """
        val = (a(tau, ella) * ((a(tau, ella) * x) ** (b(distance, el) - 1))\
               * np.exp(-1 * a(tau, ella) * x)) / gamma(b(distance, el))
        return val

    return pandel


################################################

def gauss(x, mu, sigma):
    """
    Returns a normed gaussian.

    Args:
        x (np.ndarray): x values
        mu (float): Gauss mu
        sigma (float): Gauss sigma
        n:

    Returns:

    """

    def _gaussnorm(sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi))

    return _gaussnorm(sigma) * np.exp(-np.power((x - ( mu)), 2) / (2 *  (sigma ** 2)))



################################################

def n_gauss(x, mu, sigma, n):
    """
    Returns a normed gaussian in the case of n ==1. If n > 1, The gaussian
    mean is shifted by n and its width is enlarged by the factor of n.
    The envelope of a sequence of these gaussians will be an expoenential.

    Args:
        x (np.ndarray): x values
        mu (float): Gauss mu
        sigma (float): Gauss sigma
        n (int): > 0, linear coefficient

    Returns:

    """
    assert n > 0, "Can not compute gauss with n <= 0"

    def _gaussnorm(sigma, n):
        return 1 / (sigma * np.sqrt(2 * n * np.pi))

    return _gaussnorm(sigma, n) * np.exp(-np.power((x - (n * mu)), 2) / (2 * n * (sigma ** 2)))

####################################################

def calculate_sigma_from_amp(amp):
    """
    Get the sigma for the gauss from its peak value.
    Gauss is normed

    Args:
        amp (float):

    Returns:
        float
    """
    return 1/(np.sqrt(2*np.pi)*amp)

#####################################3

def exponential(x, lmbda):
    """
    An exponential model, e.g. for a decay with coefficent lmbda.

    Args:
        x (float): input
        lmbda (float): The exponent of the exponential

    Returns:
        np.ndarray
    """
    return np.exp(-lmbda * x)

#####################################################

def calculate_chi_square(data, model_data):
    """
    Very simple estimator for goodness-of-fit. Use with care.
    Non normalized bin counts are required.

    Args:
        data (np.ndarray): observed data (bincounts)
        model_data (np.ndarray): model predictions for each bin

    Returns:
        np.ndarray
    """
    #FIXME there might be a reason I had chosen 
    #      for going with my own implenetation 
    #      initially
    mask = np.logical_and(np.isfinite(data), np.isfinite(model_data))
    chi2 =  (data[mask] - model_data[mask])**2/data[mask]
    chi2 =  (chi2[np.isfinite(chi2)].sum())
    #chi2 = st.chisquare(scale, model_data)[0]
    Logger.warn(f'Calculated chi2 of {chi2}')
    return chi2
    #return chi[np.isfinite(chi)].sum()

#################################################

def calculate_reduced_chi_square(data, model_data, sigma):
    """
    Very simple estimator for goodness-of-fit. Use with care.
    
    Args:
        data (np.ndarray)       : observed data 
        model_data (np.ndarray) : model predictions 
        sigma (np.ndarray)      : associated errors
    Returns:
    """
    #FIXME there might be a reason I had chosen 
    #      for going with my own implenetation 
    #      initially
    mask = np.logical_and(np.isfinite(data), np.isfinite(model_data), sigma > 0)
    chi2 =  (data[mask] - model_data[mask])**2/sigma[mask]
    chi2 =  (chi2[np.isfinite(chi2)].sum())
    #chi2 = st.chisquare(scale, model_data)[0]
    Logger.warn(f'Calculated chi2 of {chi2}')
    return chi2
    #return chi[np.isfinite(chi)].sum()

#################################################

def williams_correction():
    pass

#################################################

def fwhm_gauss(x, mu, fwhm, amp):
    """
    A gaussian typically used for energy spectra fits of radiotion, where
    resolutions/linewidths are typically given in full widht half maximum (fwhm)

    Args:
        x (float)    : input
        mu (float)   : peak position
        fwhm (float) : full width half maximum
        amp (float)  : amplitude
    Return:
        float        : function value
    """
    return amp*np.exp((-4*np.log(2)*((x-mu)**2))/(fwhm**2))




################################################
