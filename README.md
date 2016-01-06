# pyevsel
An API for event selection as used in HEP analysis.


Requirements
---------------------------
`dashi`
`pandas`
`root_numpy` (if root support is needed)
`yaml`

Rationale
----------------------------

This software was developed while doing an analysis for the [IceCube] (http://www.icecube.wisc.edu) Experiment.

For this analysis IceCube data has to be filtered down from 30 events/second to 10 events/year. To estimate signal efficiency and background supression power of the developed filter, the data is compared to simulated data. Different variables are exploited for the filter, and to estimate their performance, they are evalueted for experimental data and various types of simulations. 

Besides writing actual filter and analysis code, a lot of "helper" code was written. It performes tasks like reading out variables from different files and stacking them together, keeping track of their origin.
Especially keeping alignment between the different types of data (experimental or any simulation) is important. Such "helper" code is e.g. available in the root framework and can be accessed by python through [PyROOT] (https://root.cern.ch/pyroot), but a clean, lean and fun-to-use API was not available yet.

In parts, the problem has been already adressed by [dashi] (https://github.com/emiddell/dashi) which comes with a dataset/hub API, which takes care of wiring between Variables and datafiles, by providing a bundle class which basically extends the dictionary by a lean API to quickly access its contents.

Basing on this idea, the pyevsel project was created, providing basically an easy-to-use container style API for accessing many variables in different types of categories.

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

Each point in the *datacube* can be adressed by its coordinates (A,1,2,..) and then it can be searched for clustering, e.g. by machine learning methods e.g. by [skykit-learn](http://scikit-learn.org/stable/documentation.html)

The pyevsel project should provide everything to go from files to a working datacube as quickly as possible. As filtering is important from an early stage on, it will provide also a set of routines which allow filtering and cross-checking the results.

As a bonus, it allows for easy-weighing on the fly fro IceCube analysis.

##Examples


###Setting up categories from files
```
import pyevsel.variables.categories as c

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

```
variable_def.py

...
energy  = v.Variable("energy",bins=n.linspace(0,10,20),transform=n.log10,label=r"$\log(E_{rec}/$GeV$)$",definitions=[("CredoFit","energy")])

mc_p_en = v.Variable("mc_p_en",definitions=[("MCPrimary","energy"),("mostEnergeticPrimary","energy")])
mc_p_ty = v.Variable("mc_p_ty",definitions=[("MCPrimary","type"),("mostEnergeticPrimary","type")],transform=conv.ConvertPrimaryToPDG)
mc_p_ze = v.Variable("mc_p_zen",definitions=[("MCPrimary","zenith"),("mostEnergeticPrimary","zenith")],transform=conv.ConvertPrimaryToPDG)
...

```


### getting variables and weighting


```
import variable_defs
import icecube_goodies.weighting as gw
import icecube_goodies.shortcuts as s

# define the primary paramters of the simulation
signal.set_mc_primary(energy_var=variable_defs.mc_p_en,type_var=variable_defs.mc_p_ty,zenith_var=variable_defs.mc_p_ze)
signal.load_vardefs(variable_defs)
signal.read_variables()
signal.set_weightfunction(gw.GetModelWeight)
signal.get_weights(s.NuFluxes.E2)
honda.set_weightfunction(gw.GetModelWeight)
honda.get_weights(s.NuFluxes.Honda2006H3a)
```











