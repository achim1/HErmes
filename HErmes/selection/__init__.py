"""
Provides containers for in-memory variable. These containers are called "categroies",
and they represent a set of variables for a certain type of data. Categories can
be further grouped into "Datasets". Variables can be read out from files and stored
in memory in the form of numpy arrays or pandas DataSeries/DataFrames. Selection criteria
can be applied simultaniously (and reversibly) to all categories in a dataset with the "Cut"
class.

HErmes.selection provides the following submodules:

- `categories` : Container classes for variables.

- `dataset` : Grouping categories together.

- `cut` : Apply selection criteria on variables in a category.

- `variables` : Variable definition. Harvest variables from files.

- `magic_keywords` : A bunch of fixed names for automatic weight calculation.


"""
from __future__ import absolute_import

import hjson
import os
import os.path
import inspect
import importlib

from ..utils.logger import Logger

from . import categories as c
from . import dataset as ds
from ..icecube_goodies import weighting as wgt
from ..analysis import fluxes as fluxes

def load_dataset(config, variables=None):
    """
    Read a json configuration file and load a dataset populated
    with variables from the files given in the configuration file.

    Args:
        config (str/dict): json style config file or dict

    Keyword Args:
        variables (list): list of strings of variable names to read out

    Returns:
        HErmes.selection.dataset.Dataset

    """
    cfg = config
    if not isinstance(config, dict):
        assert os.path.exists(config), "Config file {} does not exist!".format(config)
        cfg = hjson.load(open(config))

    categories = dict()
    weightfunctions = dict()
    models = dict()
    model_args = dict()
    files_basepath   = cfg["files_basepath"]
    for cat in list(cfg["categories"].keys()):
        thiscat = cfg["categories"][cat]
        sanitizer = lambda x: True

        if "file_regex" in thiscat:
            to_sanitize = thiscat["file_regex"]
            def sanitizer(x):
                if to_sanitize in x:
                    return True
                else: 
                    return False

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
                                                   sanitizer=sanitizer,\
                                                   ending=thiscat["file_type"])

            #weightfunctions[cat] = dict(inspect.getmembers(wgt))[thiscat["model_method"]]
            if not "model" in thiscat:
                models[cat] = None
                model_args[cat] = [None]
            else:
                try:
                    fluxclass, flux = thiscat["model"].split(".")
                    #fluxclass = thiscat["model"]
                    #flux = "__call__"
                    models[cat] = getattr(dict(inspect.getmembers(fluxes))[fluxclass],flux)
                    model_args[cat] = thiscat["model_args"]
                except ValueError:
                    Logger.warning("{} does not seem to be a valid model for {}. This might cause troubles. If not, it is probably fine!".format(thiscat["model"],cat))
                    models[cat] = None

        elif thiscat["datatype"] == "data":
            categories[cat] = c.Data(cat)
            categories[cat].get_files(os.path.join(files_basepath,thiscat['subpath']),\
                                      prefix=thiscat["file_prefix"],\
                                      sanitizer=sanitizer,\
                                      ending=thiscat["file_type"])
            #models[cat] = float(thiscat["livetime"])
            #weightfunctions[cat] = dict(inspect.getmembers(wgt))[thiscat["model_method"]]
            
        elif thiscat["datatype"] == "reweighted":
            pass
        else:
            raise TypeError("Data type not understood. Has to be either 'simulation', 'reweighted' or 'data'!!")

    # at last we can take care of reweighted categories
    for cat in list(cfg["categories"].keys()):
        thiscat = cfg["categories"][cat]
        if thiscat["datatype"] == "reweighted":
            categories[cat] = c.ReweightedSimulation(cat,categories[thiscat["parent"]])
            #if thiscat["model"]:
            #    fluxclass, flux = thiscat["model"].split(".")
            #    models[cat] = getattr(dict(inspect.getmembers(fluxes))[fluxclass],flux)
            #    weightfunctions[cat] = dict(inspect.getmembers(wgt))[thiscat["model_method"]]
        elif thiscat["datatype"] in ["data", "simulation"]:
            pass
        else:
            raise TypeError("Data type not understood. Has to be either 'simulation', 'reweighted' or 'data'!!")

    for cat in categories:
        if isinstance(categories[cat], c.Data):
            continue
        if not "weights" in  cfg["categories"][cat]:
            continue
        categories[cat].weightvarname = cfg["categories"][cat]["weights"]

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
    #dataset.set_weightfunction(weightfunctions)
    #dataset.get_weights(models=models)
    dataset.calculate_weights(model=models, model_args=model_args)
    plot_dict = {}
    for k in cfg["categories"]:
        if "plotting" in cfg["categories"][k]:
            plot_dict[k] = cfg["categories"][k]["plotting"]
    dataset.default_plotstyles = plot_dict
    return dataset
