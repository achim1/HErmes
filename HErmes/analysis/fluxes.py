"""
Models for particle fluxes. These are just examples, for specific cosmic ray modelss
have a look at e.g. https://github.com/afedynitch/CRFluxModels.git
"""
import numpy as np

class PowerLawFlux(object):
    """
    A flux only dependent on the energy of a particle, following a power law. Defined in
    an energy interval [emin, emax] with fluence phi0 and spectral index gamma
    """

    def __init__(self, emin, emax, phi0, gamma):
        """
        Args:
            emin  (float): minimum energy
            emax  (float): maximum energy
            phi0  (float): normalization
            gamma (float): spectral index
        """
        self.emin = emin
        self.emax = emax
        self.gamma = gamma
        self.phi0 = phi0

    @staticmethod
    def E2_1E8(energy):
        """
        A flux with fixed parameters, spectral index E**-2 and normalization 1E-8
        Usefull for automatic weighting.

        Args:
            energy
        """
        return 1e-8*np.power(energy, -2)

    def __call__(self, energy):
        """
        Calculate the flux for a given energy
        Args:
            energy (np.ndarray): primary energy

        Returns:
            np.ndarray
        """
        energy = energy.astype(np.float64)

        fl = self.phi0 * np.power(energy, self.gamma)
        fl[energy < self.emin] = 0.
        fl[energy > self.emax] = 0.
        return fl

    def fluxsum(self):
        """
        The integrated flux

        Returns:
            float
        """

        if self.gamma < -1:
            ex = 1 + self.gamma
            return self.phi0 * (self.emax ** ex - self.emin ** ex) / ex
        elif self.gamma == -1:
            return self.phi0 * (np.log(self.emax) - np.log(self.emin))
        else:
            raise ValueError("Integration of positive gamma not supported")

class Constant(object):

    @staticmethod
    def identity(x):
        return np.ones(len(x))
