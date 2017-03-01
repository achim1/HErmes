"""
Provide routines for fitting charge histograms
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

from builtins import zip
from builtins import dict
from future import standard_library
standard_library.install_aliases()
import sys
import pylab as p
import numpy as np
from scipy.misc import factorial

from functools import reduce
from copy import deepcopy as copy
from collections import namedtuple

import scipy.optimize as optimize
import seaborn.apionly as sb
import dashi as d

from . import tools
from . import plotting as plt

d.visual()

try:
    from iminuit import Minuit
except ImportError:
    print ("WARNING, can not load iminuit")

# default color palette
PALETTE = sb.color_palette("dark")


def reject_outliers(data, m=2):
    """
    A simple way to remove extreme outliers from data

    Args:
        data (np.ndarray): data with outliers
        m (int): number of standard deviations outside the
                 data should be discarded

    Returns:
        np.ndarray
    """

    return data[abs(data - np.mean(data)) < m * np.std(data)]






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






def pedestal_fit(filename, nbins, fig=None):
    """
    Fit a pedestal to measured waveform data
    One shot function for
    * integrating the charges
    * making a histogram
    * fitting a simple gaussian to the pedestal
    * calculating mu
        P(hit) = (N_hit/N_all) = exp(QExCExLY)
        where P is the probability for a hit, QE is quantum efficiency,
        CE is the collection efficiency and
        LY the (unknown) light yield

    Args:
        filename (str): Name of the file with waveform data
        nbins (int): number of bins for the underlaying charge histogram

    """

    head, wf = tools.load_waveform(filename)
    charges = -1e12 * tools.integrate_wf(head, wf)
    plt.plot_waveform(head, tools.average_wf(wf))
    p.savefig(filename.replace(".npy", ".wf.pdf"))
    one_gauss = lambda x, n, y, z: n * fit.gauss(x, y, z, 1)
    ped_mod = fit.Model(one_gauss, (1000, -.1, 1))
    ped_mod.add_data(charges, nbins, normalize=False)
    ped_mod.fit_to_data(silent=True)
    fig = ped_mod.plot_result(add_parameter_text=((r"$\mu_{{ped}}$& {:4.2e}\\", 1), \
                                                  (r"$\sigma_{{ped}}$& {:4.2e}\\", 2)), \
                              xlabel=r"$Q$ [pC]", ymin=1, xmax=8, model_alpha=.2, fig=fig, ylabel="events")

    ax = fig.gca()
    n_hit = abs(ped_mod.data.bincontent - ped_mod.prediction(ped_mod.xs)).sum()
    ax.grid(1)
    bins = np.linspace(min(charges), max(charges), nbins)
    data = d.factory.hist1d(charges, bins)
    n_pedestal = ped_mod.data.stats.nentries - n_hit

    mu = -1 * np.log(n_pedestal / ped_mod.data.stats.nentries)

    print("==============")
    print("All waveforms: {:4.2f}".format(ped_mod.data.stats.nentries))
    print("HIt waveforms: {:4.2f}".format(n_hit))
    print("NoHit waveforms: {:4.2f}".format(n_pedestal))
    print("mu = -ln(N_PED/N_TRIG) = {:4.2e}".format(mu))

    ax.fill_between(ped_mod.xs, 1e-4, ped_mod.prediction(ped_mod.xs),\
                    facecolor=PALETTE[2], alpha=.2)
    p.savefig(filename.replace(".npy", ".pdf"))

    return ped_mod

################################################

def fit_model(charges, model, startparams=None, \
              rej_outliers=False, nbins=200, \
              silent=False,\
              parameter_text=((r"$\mu_{{SPE}}$& {:4.2e}\\", 5),),
              use_minuit=False,\
              normalize=True,\
              **kwargs):
    """
    Standardazied fitting routine

    Args:
        charges (np.ndarray): Charges obtained in a measurement (no histogram)
        model (pyosci.fit.Model): A model to fit to the data
        startparams (tuple): initial parameters to model, or None for first guess

    Keyword Args:
        rej_outliers (bool): Remove extreme outliers from data
        nbins (int): Number of bins
        parameter_text (tuple): will be passed to model.plot_result
        use_miniuit (bool): use minuit to minimize startparams for best 
                            chi2
        normalize (bool): normalize data before fitting
        silent (bool): silence output
    Returns:
        tuple
    """
    if rej_outliers:
        charges = reject_outliers(charges)
    if use_minuit:

        from iminuit import Minuit

        # FIXME!! This is too ugly. Minuit wants named parameters ... >.<

        assert len(startparams) > 10; "Currently more than 10 paramters are not supported for minuit fitting!"
        assert model.all_coupled, "Minuit fitting can only be done for models with all parmaters coupled!"
        names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]

        funcstring = "def do_min("
        for i,__ in enumerate(startparams):
            funcstring += names[i] + ","
        funcstring = funcstring[:-1] + "):\n"
        funcstring += "\tmodel.startparams = ("
        for i,__ in enumerate(startparams):
            funcstring += names[i] + ","
        funcstring = funcstring[:-1] + ")\n"
        funcstring += "\tmodel.fit_to_data(charges, nbins, silent=True, **kwargs)"
        funcstring += "\treturn model.chi2_ndf"


        #def do_min(a, b, c, d, e, f, g, h, i, j, k): #FIXME!!!
        #    model.startparams = (a, b, c, d, e, f, g, h, i, j, k)
        #    model.fit_to_data(charges, nbins, silent=True, **kwargs)
        #    return model.chi2_ndf
        exec(funcstring)
        bnd = kwargs["bounds"]
        if "bounds" in kwargs:
            min_kwargs = dict()
            for i,__ in enumerate(startparams):
                min_kwargs["limit_" + names[i]] =(bnd[0][i],bnd[1][i])
            m = Minuit(do_min, **min_kwargs)
            #m = Minuit(do_min, limit_a=(bnd[0][0],bnd[1][0]),
            #                   limit_b=(bnd[0][1],bnd[1][1]),
            #                   limit_c=(bnd[0][2],bnd[1][2]),
            #                   limit_d=(bnd[0][3],bnd[1][3]),
            #                   limit_e=(bnd[0][4],bnd[1][4]),
            #                   limit_f=(bnd[0][5],bnd[1][5]),
            #                   limit_g=(bnd[0][6],bnd[1][6]),
            #                   limit_h=(bnd[0][7],bnd[1][7]),
            #                   limit_i=(bnd[0][8],bnd[1][8]),
            #                   limit_j=(bnd[0][9],bnd[1][9]),
            #                   limit_k=(bnd[0][10],bnd[1][10]))
        else:



            m = Minuit(do_min)
        # hand over the startparams
        for key, value in zip(["a","b","c","d","e","f","g","h","i","j"], startparams):
            m.values[key] = value
        m.migrad()
    else:
        model.startparams = startparams
        model.add_data(charges, nbins=nbins, normalize=normalize,\
                       create_distribution=True)
        model.fit_to_data(silent=silent, **kwargs)

    # check for named tuple
    if hasattr(startparams, "_make"): # duck typing
        best_fit_params = startparams._make(model.best_fit_params)
    else:
        best_fit_params = model.best_fit_params
    print("Best fit parameters {}".format(best_fit_params))

    return model

############################################



