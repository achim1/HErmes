"""
Provide routines for fitting charge histograms
"""
import pylab as p
import numpy as np
import warnings

import dashi as d

d.visual()

try:
    from iminuit import Minuit
except ImportError:
    print ("WARNING, can not load iminuit")

def reject_outliers(data, m=2):
    """
    A simple way to remove extreme outliers from data

    Args:
        data (np.ndarray): data with outliers
        m           (int): number of standard deviations outside the
                           data should be discarded

    Returns:
        np.ndarray
    """

    return data[abs(data - np.mean(data)) < m * np.std(data)]

################################################

def fit_model(charges, model, startparams=None, \
              rej_outliers=False, nbins=200, \
              silent=False,\
              parameter_text=((r"$\mu_{{SPE}}$& {:4.2e}\\", 5),),
              use_minuit=False,\
              normalize=True,\
              **kwargs):
    """
    Standardazied fitting routine. 

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
    warnings.warn(
        "fit_model is deprecated and will go away in a future releas. Use model.fit_to_data instead.",
        DeprecationWarning
        )


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



