"""
Load variables from files into memory,
sort them into different categories whcih
represent experimental data or several
types of simulation
"""
from __future__ import absolute_import

import hjson
import os
import os.path
import inspect
import importlib

from pyevsel.utils.logger import Logger

from . import categories as c
from . import dataset as ds
from pyevsel.icecube_goodies import weighting as wgt
from pyevsel.icecube_goodies import fluxes as fluxes

def load_dataset(config, variables=None):
    """
    Loads a dataset according to a 
    configuration file
    
    Args:
        config (str/dict): json style config file or dict
    """
    cfg = config
    if not isinstance(config, dict):
        assert os.path.exists(config), "Config file {} does not exist!".format(config)
        cfg = hjson.load(open(config))

    categories = dict()
    weightfunctions = dict()
    models = dict()
    files_basepath   = cfg["files_basepath"]
    for cat in list(cfg["categories"].keys()):
        thiscat = cfg["categories"][cat]
        if thiscat["datatype"] == "simulation":
            categories[cat] = c.Simulation(cat)
            # remember that json keys are strings, so 
            # convert to int
            datasets = {}
            if "datasets" in thiscat:
                datasets = {int(x): int(thiscat['datasets'][x]) for x in thiscat['datasets']}

            categories[cat].get_files(os.path.join(files_basepath,\
                                                   thiscat['subpath']),\
                                                   prefix=thiscat["file_prefix"],\
                                                   datasets=datasets,\
                                                   ending=thiscat["file_type"])

            weightfunctions[cat] = dict(inspect.getmembers(wgt))[thiscat["model_method"]]
            if "constant" in thiscat["model_method"]:
                models[cat] = float(thiscat["model"])
            else:
                try:
                    fluxclass, flux = thiscat["model"].split(".")
                    models[cat] = getattr(dict(inspect.getmembers(fluxes))[fluxclass],flux)
                except ValueError:
                    Logger.warning("{} does not seem to be a valid model for {}. This might cause troubles. If not, it is probably fine!".format(thiscat["model"],cat))
                    models[cat] = None

        elif thiscat["datatype"] == "data":
            categories[cat] = c.Data(cat)
            categories[cat].get_files(os.path.join(files_basepath,thiscat['subpath']),\
                                      prefix=thiscat["file_prefix"],\
                                      ending=thiscat["file_type"])
            models[cat] = float(thiscat["livetime"])
            weightfunctions[cat] = dict(inspect.getmembers(wgt))[thiscat["model_method"]] 
            
        elif thiscat["datatype"] == "reweighted":
            pass
        else:
            raise TypeError("Data type not understood. Has to be either 'simulation', 'reweighted' or 'data'!!")
    # at last we can take care of reweighted categories
    for cat in list(cfg["categories"].keys()):
        thiscat = cfg["categories"][cat]
        if thiscat["datatype"] == "reweighted":
            categories[cat] = c.ReweightedSimulation(cat,categories[thiscat["parent"]])
            if thiscat["model"]:
                fluxclass, flux = thiscat["model"].split(".")
                models[cat] = getattr(dict(inspect.getmembers(fluxes))[fluxclass],flux)
                weightfunctions[cat] = dict(inspect.getmembers(wgt))[thiscat["model_method"]] 
        elif thiscat["datatype"] in ["data","simulation"]:
            pass
        else:
            raise TypeError("Data type not understood. Has to be either 'simulation', 'reweighted' or 'data'!!")

    #combined categories
    combined_categories = dict() 
    for k in list(combined_categories.keys()):
        combined_categories[k] = [categories[l] for l in cfg["combined_categories"]]

    # import variable defs
    vardefs = cfg["variable_definitions"]
    if isinstance(vardefs, str) or isinstance(vardefs, unicode):
        vardefs = importlib.import_module(cfg["variable_definitions"])
    elif isinstance(vardefs, dict):
        vardefs = {}
        for k in cfg["variable_definitions"]:
            vardefs[k] = importlib.import_module(cfg["variable_definitions"][k])
    else:
        raise ValueError("Can not understand variable definitions {} of type {}".\
                         format(vardefs, type(vardefs)))
    #vardefs = importlib.import_module(cfg["variable_definitions"])
    dataset = ds.Dataset(*list(categories.values()),\
                         combined_categories=combined_categories)

    dataset.load_vardefs(vardefs)
    dataset.read_variables(names=variables)
    dataset.set_weightfunction(weightfunctions)
    dataset.get_weights(models=models)
    plot_dict = {}
    for k in cfg["categories"]:
        if "plotting" in cfg["categories"][k]:
            plot_dict[k] = cfg["categories"][k]["plotting"]
    dataset.default_plotstyles = plot_dict
    return dataset
