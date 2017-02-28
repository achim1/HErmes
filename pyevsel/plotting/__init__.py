"""
Provide the usual plots
"""
import tempfile
try:
    import yaml
except ImportError:
    import ruamel.yaml as yaml
import os
import re

from pyevsel.utils.logger import Logger
STD_CONF=os.path.join(os.path.split(__file__)[0],"plotsconfig.yaml")

# write configuration to tmpfile
config = tempfile.NamedTemporaryFile(prefix="plotcfg",delete=False)
CONFIGFILE = config.name
SLASHES = re.compile(r"(\\+)")

with open(STD_CONF,"r") as configfile:
    configdata = configfile.read()
    try:
        config.write(configdata)
    except TypeError:
        config.write(bytes(configdata.encode()))
    config.close()

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
    with open(configfile,"r") as cfgfile:
        data = cfgfile.read()

    with  open(CONFIGFILE,"w") as config:
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
            # FIXME little hack for bad latex parsing
            # by yaml
            #cleanlabel = cfg["label"]
            cleanlabel = SLASHES.sub(r"\\",cfg["label"])
            cfg["label"] = cleanlabel
            return cfg
    Logger.warning("No config for {0} found!".format(name))
    return cfg

########################################

def get_config_item(key,filename=STD_CONF):
    """
    Returns an item from upper level tree in the config

    Args:
        item (str): the key in the config file

    Returns:
        str: config setting
    """
    cfg = yaml.load(open(filename,"r"))
    return cfg[key]

########################################

def PrintConfig():
    """
    Print a representation of the configfile
    """
    config = LoadConfig()
    return yaml.dump( config, default_flow_style=False, default_style='' )

############################################
