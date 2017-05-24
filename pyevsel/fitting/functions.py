"""
Provide some simple functions which can be used to create models
"""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from scipy.misc import factorial
from future import standard_library
standard_library.install_aliases()


def poisson(k, lmbda):
    """
    Poisson distribution

    Args:
        lmbda (int): expected number of occurences
        k (int): measured number of occurences

    Returns:
        np.ndarrya
    """

    return np.power(lmbda, k) * np.exp(-1 * lmbda) / factorial(k)

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


def exponential(x, N, n_noise):
    """
    An exponential model (for noise?)

    Args:
        x:
        N:
        n_noise:

    Returns:
        np.ndarray
    """
    return N * np.exp(-n_noise * x)

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

    chi = ((data - model_data)**2/data)
    return chi[np.isfinite(chi)].sum()




