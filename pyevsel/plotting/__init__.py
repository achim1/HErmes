"""
Provide the usual plots
"""
import tempfile
import yaml
import os

from pyevsel.utils.logger import Logger
STD_CONF=os.path.join(os.path.split(__file__)[0],"plotsconfig.yaml")

# write configuration to tmpfile
config = tempfile.NamedTemporaryFile(prefix="plotcfg",delete=False)
configdata = open(STD_CONF,"r").read()
config.write(configdata)
config.close()
CONFIGFILE = config.name


#################################

def SetConfig(configdict):
    """

    Args:
        configdict (dict): A valid yaml configuration

    Returns:
        None
    """
    data = yaml.dump(configdict,open(CONFIGFILE,"w"))
    SetConfigFile(CONFIGFILE)
    return

#################################

def SetDefaultConfig():
    """
    Set the config to the default values
    """
    SetConfigFile(STD_CONF)
    return None


#################################

def SetConfigFile(configfile):
    """
    Set a new config

    Args:
        configfile: The new yaml style configuration file

    Returns:
        None
    """
    data = open(configfile,"r")
    data = data.read()
    config = open(CONFIGFILE,"w")
    config.write(data)
    config.close()
    return None

######################################

def LoadConfig():
    """
    Load a configuration from a yaml configfile
    Args:
        configfile (str): filename of file containing yaml fro configuration

    Returns:
        dict: configuration
    """
    configs = yaml.load(open(CONFIGFILE,"r"))
    return configs

######################################

def GetCategoryConfig(name):
    """
    Get the relevant config section from the actual
    config for a category

    Args:
        name (string): Name of a category to search for
    """

    configs = yaml.load(open(CONFIGFILE,"r"))
    for cfg in configs["categories"]:
        if cfg["name"] == name:
            return cfg
    Logger.warning("No config for %s found!" %name)
    return cfg

########################################

def PrintConfig():
    """
    Print a representation of the configfile
    """
    config = LoadConfig()
    return yaml.dump( config, default_flow_style=False, default_style='' )

############################################
