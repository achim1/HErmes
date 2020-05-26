[![Coverage Status](https://coveralls.io/repos/github/achim1/HErmes/badge.svg?branch=master)](https://coveralls.io/github/achim1/HErmes?branch=master)
[![Python 3.6+](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Build Status](https://travis-ci.org/achim1/HErmes.svg?branch=develop)](https://travis-ci.org/achim1/HErmes)
[![Docs](https://readthedocs.org/projects/hermes-python/badge/?version=latest)](http://hermes-python.readthedocs.io/en/latest/?badge=latest)

# HErmes

*Highly efficient rapid multipurpose event selection toolset with a focus on high energy physics analysis tasks* 

Rationale
----------------------------
Aggregate data from hdf or root files and conveniently apply filter criterias.

Background story
----------------------------

This software was developed while doing an analysis for the [IceCube] (http://www.icecube.wisc.edu) Experiment.

For this analysis data has to be filtered down from 30 events/second to 10 events/year. To estimate signal efficiency and background supression power of the developed filter, the data is compared to simulated data. Different variables are exploited for the filter, and to estimate their performance, they are evalueted for experimental data and various types of simulations. 

Besides writing actual filter and analysis code, a lot of "helper" code was written. It performes tasks like reading out variables from different files and stacking them together, keeping track of their origin.
Especially keeping alignment between the different types of data (experimental or any simulation) is important.
Such "helper" code is e.g. available in the root framework and can be accessed by python through [PyROOT] (https://root.cern.ch/pyroot), but a clean, lean and fun-to-use API was not available yet (at least in my opinion).

In parts, the problem has been already adressed by [dashi] (https://github.com/emiddell/dashi) which comes with a dataset/hub API, which takes care of wiring between Variables and datafiles, by providing a bundle class which basically extends the dictionary by a lean API to quickly access its contents.

Basing on this idea, the HErmes project was created, providing basically an easy-to-use container style API for accessing many variables in different types of categories.

The idea is quickly sketcheted as follows:

Assume we have two variables A and B (which might be stored in hdf files for example), and two categories of data 1 and 2 (which might be signal and background simulation). This leads to the following abstract situation:

```
Variable A - Category 1
Variable A - Category 2
Variable B - Category 1
Variable B - Category 2
```

To get the data, the following functions are desirable:

```
1. GetVariable(Category) -> A,B
2. GetCategory(Variable) -> 1,2
```

The first function yields all the variables from a certain category.
The second is important for developing filters, as it yields all categories for this variable, and thus e.g signal-to-noise ratios can be comparted.
In the end, what is really desired is a *datacube*, which is the full parameter space:

```
    A  B 

    1  1

    2  2 
```

Each point in the *datacube* can be adressed by its coordinates (A,1,2,..) and then it can be searched for clustering, e.g. by machine learning methods e.g. by [scikit-learn](http://scikit-learn.org/stable/documentation.html)

The HErmes project provides everything to go from files to a working datacube as quickly as possible. As filtering is important from an early stage on, it provides also a set of routines which allow filtering and cross-checking the results.

As a bonus, it allows for easy-weighting on the fly for IceCube analysis.

#Examples

##Shortcut - the `load_dataset` routine

Define a `.json file` with the following layout:

```python
{
// define files and paths 
// and names for the different
// data
// file can contain comments 
//
// datatype can be either "simulation","reweighted", or 'data'
  "files_basepath": "/home/my/data/",
  "variable_definitions": "vardefs",
  "categories": {"exp": {
                        "datatype": "simulation",
                        "subpath": "sim",
                        "file_prefix": "",
                        "file_type": ".h5",
                        "model_method" : "constant_weights",
                        "model" : 1,
                        "plotting": {"label": "'$\nu_{astr}$'",
                                     "linestyle": {"color": 2,
                                                    "linewidth": 3,
                                                    "alpha": 1,
                                                    "filled": 0,
                                                    "linestyle": "solid"},
                                     "histotype": "line"}
                     }
                "sim": {....
                .....} 
                }
}
```

The files have to be in subfolders of `/home/my/data` in this case and variables have
to be defined in a python file (see below) in the $PYTHONPATH with the name `vardefs.py`.

The filename of the config file can then be given to `HErmes.selection.load_dataset` which will return a dataset (see below)

##Setting up categories from files

First tell the software, where to find the files to read the data from. Categories need to be initialized with an unique name.
The ReweightedSimulation category holds a reference to all the variables defined in the given category, however it allows for 
the calculation of different weights.

```python
import HErmes.selection.categories as c

numu_path = "/some/path"
signal = c.Simulation("astro_numu")
# Reweighted Simulation holds a reference to
# a given category, but allows for different
# weights
honda  = c.ReweightedSimulation("conv_numu",signal)
signal.get_files(numu_path,ending=".h5",prefix="dccap://")
```
### defining variabls
 defining labels, bins,
 transformations can be applied...

The variable definitions can be written to its own python file, e.g. `variable_def.py` (which should be in the PYTHONPATH).
Variables are declared by a unique name and optional bins, transformations and labels. The definitions describe the table and subnode
where the variables can be found in the datafiles

```python
variable_def.py

...
energy  = v.Variable("energy",bins=n.linspace(0,10,20),transform=n.log10,label=r"$\log(E_{rec}/$GeV$)$",definitions=[("MyReco","energy")])

mc_p_en = v.Variable("mc_p_en",definitions=[("MCPrimary","energy"),("mostEnergeticPrimary","energy")])
mc_p_ty = v.Variable("mc_p_ty",definitions=[("MCPrimary","type"),("mostEnergeticPrimary","type")],transform=conv.ConvertPrimaryToPDG)
mc_p_ze = v.Variable("mc_p_zen",definitions=[("MCPrimary","zenith"),("mostEnergeticPrimary","zenith")],transform=conv.ConvertPrimaryToPDG)
...

```


### getting variables and weighting

The weighting which is provided with the module is specific for IceCube and needs the icetray software. However, it could be extended easily to other experiments.


### cutting

A cut can be defined by giving conditions with simple strings. After application, the get calls of the category will return
the variable data after applying the cut.
If necessary, cuts can be deleted

```python
uncut_var  =  signal.get("myfavoritevariable")
cut = cu.Cut(variables=[("energy",">",5)])
uncutted_var = signal.get("myfavoritevariable")
signal.add_cut(cut)
signal.apply_cuts()
cutted_var = signal.get("myfavoritevariable")
print len(uncutted_var) > len(cutted_var)
=> True
```

#### undo and delete cuts

A cut can be undone (if not the inplace parameter is set to true) with `Category.undo_cuts()` (which is the inverse to `Category.apply_cuts()`) and completely removed with `Category.delete_cuts()`


### datasets

A dataset is a combination of different cutegories, which allows to perform cuts simultaniously on all categories of the
dataset

```python
import HErmes.selection.cut as cu

# background, data are of type HErmes.selection.categories.Category as well
dataset = c.Dataset(signal,background,data)
cut = cu.Cut(("energy",">",5))
another_cut = cu.Cut(("zenith", "<=", 60))

# cuts can be added
allcuts = cut + another_cut

dataset.add_cut(allcuts)
dataset.apply_cuts()
```

#### gotchas

**easy plotting of variable distributions**:

* plots the distribution with ratio and cumulative distribution the `ratio` parameter defines which should be plot in the ratio part height defines the fraction of the individual panels (*The API for the ratio parameter might change slightly*)

```python

dataset.plot_distribution("energy",ratio=([data.name],[background.name]),heights=[.3,.2,.2])
```




**rate table**: 

* if weights are avaiable, the integrated rates can be displayed in a nice html table representation (uses dashi.tinytable.TinyTable). Signal and background parameters allow to calculate sum rates for signal and background

```python
table = dataset.tinytable(signal=[signal],background=[background])
```







