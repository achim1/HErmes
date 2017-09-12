import pytest
import numpy as np
import pandas as pd
import os
import hjson
import matplotlib
matplotlib.use("Agg")

import HErmes.selection.categories as cat
import HErmes.selection.dataset as ds
import HErmes.selection.variables as v
import HErmes.selection.magic_keywords as mk


from HErmes.selection import load_dataset

import testvardefs
from fixturefactory import prepare_testtable, prepare_sparser_testtable, TESTDATALEN 

# test the magic keywords ..
from HErmes.selection import magic_keywords
from HErmes.selection import cut


# define some test variables
# run and event id
V = v.Variable
run   = V("RUN",definitions=[("I3EventHeader","Run")])
event = V("EVENT",definitions=[("I3EventHeader","Run")])

# mc primary
# the variable names are magic!
mc_p_en     = V("mc_p_en", definitions=[("MCPrimary","energy"),("mostEnergeticPrimary","energy")])
mc_p_ty     = V("mc_p_ty", definitions=[("MCPrimary","type"),("mostEnergeticPrimary","type")])
mc_p_ze     = V("mc_p_ze", definitions=[("MCPrimary","zenith"),("mostEnergeticPrimary","zenith")], transform=np.cos)
mc_p_we     = V("mc_p_we", definitions=[("I3MCWeightDict","TotalInteractionProbabilityWeight"),("CorsikaWeightMap","DiplopiaWeight")])
mc_p_gw     = V('mc_p_gw', definitions=[('CorsikaWeightMap','Weight')])
mc_p_ts     = V('mc_p_ts', definitions=[('CorsikaWeightMap','TimeScale')])

# more MC related things
mc_nevents  = V("mc_nevents", definitions=[("I3MCWeightDict","NEvents")])


def hobo_weightfunc(*args, **kwargs):
    return np.ones(TESTDATALEN)

def generate_test_cuts():
    return [("mc_p_en", ">=", 5), ("charge","<=", 10),\
            ("energy",">", 10), ("velocity", "<", 42 )]

def generate_test_cut_variablenames():
    return [k[0] for k in generate_test_cuts()]


def test_Variable(prepare_testtable):
    testvar = mc_p_en
    testvar.declare_harvested()
    assert testvar.is_harvested
    testvar.undeclare_harvested()
    assert (not testvar.is_harvested)
    assert (testvar == testvar)
    testvar.harvest(str(prepare_testtable.realpath()))
    testvar.declare_harvested()
    testvar.calculate_fd_bins()
    testvar.rewire_variables({"foo" : "bar"})
    assert (len(testvar.bins) > 0)
    assert isinstance(hash(testvar), int)

    comp_var = v.CompoundVariable("comb_energy", variables=[testvar, testvar])
    assert (comp_var < testvar)
    assert isinstance(hash(comp_var), int)
    assert isinstance(comp_var.__repr__(), str)
    comp_var.harvest(prepare_testtable.realpath())
    assert comp_var.is_harvested
    comp_var.rewire_variables(dict(zip([i.name for i in comp_var.variables ], comp_var.variables)))


    list_var = v.VariableList("list_energy", variables=[testvar, testvar])
    assert isinstance(hash(list_var), int)
    assert isinstance(list_var.__repr__(), str)
    list_var.harvest(prepare_testtable.realpath())
    assert list_var.is_harvested
    assert len(list_var.data) == 2
    list_var.rewire_variables(dict(zip([i.name for i in list_var.variables],list_var.variables)))

    testvar > comp_var
    list_var < comp_var
    assert list_var == list_var
    assert list_var >= list_var
    assert comp_var <= comp_var

def test_Cut():
    testcut = cut.Cut(*generate_test_cuts())
    assert sorted(testcut.variablenames) == sorted(generate_test_cut_variablenames())
    assert len([k for k in testcut]) == len(generate_test_cut_variablenames())
    assert isinstance(testcut.__repr__(), str)
    testcut2 = cut.Cut(*generate_test_cuts())
    testcut2 += testcut
    assert len(testcut.variablenames) == len(testcut2.variablenames)
    


def test_init_Cut_w_condition():
    testcut = cut.Cut(*generate_test_cuts(), name="Test", condition=np.ones(20))
    assert sorted(testcut.variablenames) == sorted(generate_test_cut_variablenames())
    assert len([k for k in testcut]) == len(generate_test_cut_variablenames())
    assert isinstance(testcut.__repr__(), str)


def test_simcat(prepare_testtable):
    import numpy as np
    import os.path

    sim = cat.Simulation("neutrino")
    assert not sim.show()
    assert sim.explore_files() is None
    assert hash(sim) == hash((sim.name,""))
    assert sim == sim
    assert len(sim) == 0

    filename = str(prepare_testtable.realpath())
    sim.get_files(os.path.split(filename)[0], ending=".h5", prefix="", sanitizer=lambda x : "test" in x)
    sim.get_files(os.path.split(filename)[0], use_ls = True, ending=".h5", prefix="", sanitizer=lambda x : "test" in x, force=True)
    assert isinstance(sim.explore_files(), list)
    assert len(sim.files) > 0

    energy = v.Variable("energy",bins=np.linspace(0,10,20),\
                         label=r"$\log(E_{rec}/$GeV$)$",\
                         definitions=[("readout","energy")])
    #print (energy)
    sim.add_variable(energy)
    for var in [mc_p_en, mc_p_ty, mc_p_ze, mc_p_we, mc_p_gw, mc_p_ts]:
        sim.add_variable(var)

    sim.read_variables()
    assert (sim["energy"] == sim.get("energy")).all()
    lengths = sim.show() 
    assert isinstance(lengths, dict)
    assert sim.is_harvested
    assert sim.mc_p_readout
    sim.get_files(os.path.split(filename)[0], use_ls = True, ending=".h5", prefix="", sanitizer=lambda x : "test" in x, force=True)
    sim.add_variable(energy)
    for var in [mc_p_en, mc_p_ty, mc_p_ze, mc_p_we, mc_p_gw, mc_p_ts]:
        sim.add_variable(var)

    sim.read_variables()
    

    assert len(sim.get("energy")) > 0
    assert len(sim) > 0
    simlenuncut = len(sim)
    assert "energy" in sim.variablenames
    def wf(x,y,**kwargs):
        return np.ones(TESTDATALEN)
    sim.set_weightfunction(wf)
    sim.get_weights(lambda x: x)
    #assert np.isfinite(sim.livetime)
    assert sim.livetime == 1.
    energycut = cut.Cut(("energy", ">", 500**2))


    sim.add_cut(energycut)
    sim.apply_cuts()
    cutted_len = len(sim.get("energy"))
    fig = sim.distribution("energy")
    assert isinstance(fig, matplotlib.figure.Figure)
    fig = sim.distribution("energy", log=True)
    fig = sim.distribution("energy", style="scatter")
    

    sim.undo_cuts()
    sim.delete_cuts()
    assert len(sim.get("energy")) == simlenuncut
    condition = np.ones(simlenuncut)
    condition[2000] = 0
    
    energycut = cut.Cut(("energy", ">", 500**2), condition = {sim.name :condition})
    #print len (sim)
    
    sim.add_cut(energycut)
    sim.apply_cuts()
    cond_len = len(sim.get("energy"))
    assert cond_len == cutted_len + 1
    sim.undo_cuts()
    sim.delete_cuts()
    sim.add_cut(energycut)
    sim.apply_cuts(inplace=True)
    #assert False
    #FIXME len problem!
    sim.add_plotoptions({}) 


def tset_reweighted_simcat(prepare_testtable):
    import numpy as np
    import os.path

    sim = cat.Simulation("neutrino")
    filename = str(prepare_testtable.realpath())
    sim.get_files(os.path.split(filename)[0], ending=".h5", prefix="", sanitizer=lambda x : "test" in x)
    energy = v.Variable("energy",bins=np.linspace(0,10,20),\
                         label=r"$\log(E_{rec}/$GeV$)$",\
                         definitions=[("readout","energy")])
    #print (energy)
    sim.add_variable(energy)
    for var in [mc_p_en, mc_p_ty, mc_p_ze, mc_p_we, mc_p_gw, mc_p_ts]:
        sim.add_variable(var)

    sim.read_variables()
    rsim = cat.ReweightedSimulation("conv_nue", sim)
    assert rsim.raw_count == sim.raw_count
    assert (rsim.get("energy") == sim.get("energy")).all()
    rsim.rewire_variables()

    

def test_simcat_mcreadout(prepare_testtable):
    import numpy as np
    import os.path

    sim = cat.Simulation("neutrino")
    filename = str(prepare_testtable.realpath())
    sim.get_files(os.path.split(filename)[0], ending=".h5", prefix="", sanitizer=lambda x : "test" in x)
    assert len(sim.files) > 0
    for var in [mc_p_en, mc_p_ty, mc_p_ze, mc_p_we, mc_p_gw, mc_p_ts]:
        sim.add_variable(var)
    sim.read_mc_primary()
    assert sim.mc_p_readout
    assert len(sim) > 0

def test_datcat(prepare_testtable):
    import numpy as np
    import os.path

    exp = cat.Data("exp")
    assert np.isnan(exp.livetime)
    filename = str(prepare_testtable.realpath())
    exp.get_files(os.path.split(filename)[0], ending=".h5", prefix="", sanitizer=lambda x : "test" in x)
    assert len(exp.files) > 0
    energy = v.Variable("energy",bins=np.linspace(0,10,20),\
                         label=r"$\log(E_{rec}/$GeV$)$",\
                         definitions=[("readout","energy")])
    #print (energy)
    exp.add_variable(energy)
    exp.read_variables()
    assert len(exp.get("energy")) > 0
    exp.get_weights(100.)
    assert (exp.weights == (np.ones(exp.raw_count)/100.)).all()

    exp = cat.Data("exp")
    exp.get_files(os.path.split(filename)[0], ending=".h5", prefix="", sanitizer=lambda x : "test" in x)
    exp.load_vardefs(testvardefs)
    exp.read_variables()
    exp.set_run_start_stop(runstart_var="START", runstop_var="STOP")
    exp.estimate_livetime() 

     

def test_dataset(prepare_testtable, prepare_sparser_testtable):
    import os.path

    exp = cat.Data("exp")
    sim = cat.Simulation("nu")
    filename = str(prepare_testtable.realpath())
    sparser_filename = str(prepare_sparser_testtable.realpath())
    exp.get_files(os.path.split(filename)[0], ending=".h5", prefix="", sanitizer=lambda x: not "sparse" in x)
    sim.get_files(os.path.split(sparser_filename)[0], ending=".h5", prefix="", sanitizer=lambda x: "sparse" in x)

    sim2 = cat.ReweightedSimulation("conv_nu", sim)
    data = ds.Dataset(exp, sim)
    assert len(data.categorynames) == 2
    assert len(data.categories) == 2


    # test helpers
    assert ds.get_label(exp) == "exp"
    exp.plot_options = {"label" : "foo"}
    assert ds.get_label(exp) == "foo" 

    data.add_category(sim2)
    assert isinstance(data.__repr__(), str)
    assert len(data.categorynames) == 3
    assert len(data.categories) == 3
    assert len(exp.files) > 0
    with pytest.raises(KeyError) as e_info:
        data.get_category("blubb")

    data = ds.Dataset(exp, sim, sim2)
    data.load_vardefs(testvardefs)
    data.read_variables()
    assert "energy" in data.get_category("exp").vardict
    assert len(data.get_category("exp").vardict["energy"].data) > 0
    assert len(data.get_category("nu").vardict["energy"].data) > 0
    assert data["energy"].equals(data.get_variable("energy"))
    assert data["exp"] == data.get_category("exp")
    assert (data["exp:energy"] == data.get_category("exp").get("energy")).all()

    data.delete_variable("energy")
    assert "energy" not in data.get_category("exp").vardict
    assert isinstance(sim.get_datacube(), pd.DataFrame)
    del data
    data = ds.Dataset(exp, sim, sim2)
    data.load_vardefs({"exp" : testvardefs, "nu" : testvardefs})
    data.read_variables()
    assert len(data.get_category("exp").vardict["energy"].data) > 0
    assert len(data.get_category("nu").vardict["energy"].data) > 0


    energy = v.Variable("energy",bins=np.linspace(0,10,20),\
                         label=r"$\log(E_{rec}/$GeV$)$",\
                         definitions=[("readout","energy")])
    sparse_energy = v.Variable("sparseenergy",bins=np.linspace(0,10,20),\
                         label=r"$\log(E_{rec}/$GeV$)$",\
                         definitions=[("FailedReco","energy")])
    
    

    data.add_variable(energy)
    data.add_variable(mc_p_en)
    data.add_variable(mc_p_ze)
    data.add_variable(mc_p_we)
    data.add_variable(mc_p_ty)

    data.read_variables()
    assert len(data.get_category("exp").vardict["energy"].data) > 0
    assert len(data.get_category("nu").vardict["energy"].data) > 0

    assert len(data.get_category("nu")) > 0
    assert data.get_category("exp") == exp
    sparsest = data.get_sparsest_category()
    assert sparsest == "nu" 
    sparsest = data.get_sparsest_category(omit_zeros=False)
    assert sparsest == "nu" 
    allenergy = data.get_variable("energy")
    assert isinstance(allenergy, pd.DataFrame) 
    # continue here
    def model(*args, **kwargs):
        return None
    models = {"nu" : model}
    data.set_livetime(1000.)
    data.set_weightfunction(hobo_weightfunc)
    data.get_weights(models)
    assert len(data.weights) > 0
    data.integrated_rate
    data.sum_rate(categories=["exp", "nu"])
    data.calc_ratio(nominator=["exp"], denominator=["nu"])
    #data.plot_distribution("energy")
    combi_cat = cat.CombinedCategory("combined", data.categories)
    combi_cat.weights
    assert isinstance(combi_cat.vardict, dict)
    assert len(combi_cat.get("energy")) == sum([len(x) for x in data.categories])
    combi_cat.integrated_rate
    combi_cat.add_plotoptions({})

def test_load_dataset(prepare_testtable):
    filepath = str(prepare_testtable.realpath())
    filepath = os.path.split(filepath)[0]
    thisdir = os.path.split(__file__)[0]
    config = os.path.join(thisdir,"testconfig.json")

    config = hjson.load(open(config))

    #tweak the config to make the path right
    config["categories"]["neutrinos"]["subpath"] = filepath
    config["categories"]["muons"]["subpath"] = filepath
    config["categories"]["data"]["subpath"] = filepath

    data = load_dataset(config)
    assert data.weights[1]["neutrinos"] == 1

