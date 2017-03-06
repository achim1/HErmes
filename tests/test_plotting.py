import pytest
import pylab as p

import pyevsel.plotting as plt
from pyevsel.plotting import plotcolors as pc
from pyevsel.plotting import canvases as cv

def test_plotting_config_schmagoigle():
    # initialize config, search and destroy
    # FIXME
    plt.SetDefaultConfig()
    config = plt.LoadConfig()
    assert config is not None
    plt.SetConfig(plt.SetConfig(config))
    plt.SetConfigFile(plt.CONFIGFILE)
    assert isinstance(plt.PrintConfig(), str)
    for key in config.keys():
        plt.get_config_item(key)
    catconf = plt.GetCategoryConfig("atmos_mu")
    assert catconf is not None

def colordictfactory():
    testdict = {1 : "red", 2: "blue"}
    return testdict
    assert config is not None 

def test_plotcolors_ColorDict():
    cd = pc.ColorDict()
    testdict = colordictfactory()
    cd.update(testdict)
    assert cd[1] == "red"
    assert cd["nan"] == "nan"

def test_plotcolors_get_color_palette():
    cd = pc.get_color_palette()
    assert isinstance(cd, pc.ColorDict)

def test_init_YStackedCanvas():
    canvas = cv.YStackedCanvas()
    assert isinstance(canvas.figure, p.Figure)

