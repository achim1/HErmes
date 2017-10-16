import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")


import HErmes.selection.cut as cut
import HErmes.plotting as plt
import HErmes.plotting.plotting as pltplt

from HErmes.plotting import colors as pc
from HErmes.plotting import canvases as cv
from HErmes.plotting import layout


@pytest.fixture(scope='session')
def png_file(tmpdir_factory):
    png = tmpdir_factory.mktemp('data').join("testplot.png")
    return png

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
    pc.seaborn_loaded = False
    cd = pc.get_color_palette()
    assert isinstance(cd, pc.ColorDict)

def test_YStackedCanvas(png_file):
    import pylab as p
    import os.path

    canvas = cv.YStackedCanvas(figsize=layout.FIGSIZE_A4_LANDSCAPE)
    assert isinstance(canvas.figure, matplotlib.figure.Figure)
    canvas = cv.YStackedCanvas()
    assert isinstance(canvas.figure, matplotlib.figure.Figure)
    xlim = (1, 10)
    ylim = (10, 100)
    canvas.limit_xrange(*xlim)
    assert canvas.axes[0].get_xlim() == pytest.approx(xlim, abs=1e-2)
    canvas.limit_yrange(*ylim)
    assert canvas.axes[0].get_ylim() == pytest.approx(ylim, abs=1e-2)
    canvas.eliminate_lower_yticks()
    canvas.global_legend()
    pngfilename, path = os.path.split(str(png_file.realpath()))
    canvas.save(path, pngfilename)
    assert os.path.exists(canvas.png_filename)
    canvas.show()


def test_create_arrow():
    import pylab as p
    import matplotlib

    fig = p.Figure()
    ax = fig.gca()
    ax = pltplt.create_arrow(ax, 1, 1, .2, .2, 5,\
                 width = .1, shape="right",\
                 fc="k", ec="k",\
                 alpha=1., log=False)
    assert len(ax.artists) == 1
    assert isinstance(ax.artists[0], matplotlib.patches.FancyArrow)

    fig = p.figure()
    ax = fig.gca()
    ax = pltplt.create_arrow(ax, 1, 1, .2, .2, 5,\
                 width = .1, shape="left",\
                 fc="k", ec="k",\
                 alpha=1., log=True)
    assert len(ax.artists) == 1
    assert isinstance(ax.artists[0], matplotlib.patches.FancyArrow)


def test_VariableDistributionPlot():
    vplot = pltplt.VariableDistributionPlot()
    vplot.add_data(np.ones(1000), "test1", 100)
    vplot.add_cumul("test1")
    assert vplot.name == "test1"
    assert "test1" in vplot.histograms
    assert "test1" in vplot.cumuls
    vplot.add_data(np.ones(1000), "test2", 100, weights=np.ones(1000))
    vplot.add_cumul("test2")
    assert vplot.name == "test2"
    assert "test2" in vplot.histograms
    assert "test2" in vplot.cumuls
    rationame = vplot.add_ratio("test1", "test2")
    assert isinstance(rationame, str)
    testcut = cut.Cut(("test1", ">", .2))
    vplot.add_cuts(testcut)
    vplot.plot()

    
    



