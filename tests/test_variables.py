import pytest
import numpy as np

import pyevsel.variables.categories as cat
import pyevsel.variables.dataset as ds
import pyevsel.variables.variables as v
import pyevsel.variables.magic_keywords as mk

# test the magic keywords ..
from pyevsel.variables import magic_keywords

from pyevsel.variables import cut


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

TESTDATALEN = 10000



def generate_test_cuts():
    return [("mc_p_en", ">=", 5), ("charge","<=", 10),\
            ("energy",">", 10), ("velocity", "<", 42 )]

def generate_test_cut_variablenames():
    return [k[0] for k in generate_test_cuts()]

@pytest.fixture(scope='session')
def prepare_testtable(tmpdir_factory):
    from tables import IsDescription, Float64Col, open_file
    import numpy as np

    class TestParticle(IsDescription):
        energy = Float64Col()  # double (double-precision)
        x = Float64Col()

    class MCPrimary(IsDescription):
        energy = Float64Col()
        type = Float64Col()
        zenith = Float64Col()

    class CorsikaWeightMap(IsDescription):
        DiplopiaWeight = Float64Col()
        Weight = Float64Col()
        TimeScale = Float64Col()


    data = np.random.normal(5,10, TESTDATALEN)
    testdatafile = tmpdir_factory.mktemp('data').join("testdata.h5")
    testfile = open_file(str(testdatafile.realpath()), mode="w")
    #group = testfile.create_group("/", 'detector', 'Detector information')
    table = testfile.create_table("/", 'readout', TestParticle, "Readout example")
    particle = table.row
    for i in range(TESTDATALEN):
        particle['x'] = float(data[i])
        particle['energy'] = float(100.*data[i]**2)
        particle.append()
    table.flush()
    table = testfile.create_table("/", 'MCPrimary', MCPrimary, "mc primary example")
    particle = table.row
    for i in range(TESTDATALEN):
        particle['type'] = float(data[i])
        particle['energy'] = float(100. * data[i] ** 2)
        particle['zenith'] = float(100. * data[i] ** 2)
        particle.append()
    table.flush()
    table = testfile.create_table("/", 'CorsikaWeightMap', CorsikaWeightMap, "cwm example")
    particle = table.row
    for i in range(TESTDATALEN):
        particle['DiplopiaWeight'] = float(data[i])
        particle['Weight'] = float(data[i])
        particle['TimeScale'] = float(100. * data[i] ** 2)
        particle.append()
    table.flush()
    testfile.close()
    return testdatafile

def test_Variable(prepare_testtable):
    testvar = mc_p_en
    testvar.declare_harvested()
    assert testvar.is_harvested
    testvar.undeclare_harvested()
    assert (not testvar.is_harvested)
    assert (testvar == testvar)
    testvar.data = v.harvest([str(prepare_testtable.realpath())], testvar.definitions)
    testvar.declare_harvested()
    testvar.calculate_fd_bins()
    assert (len(testvar.bins) > 0)
    assert isinstance(hash(testvar), int)

    comp_var = v.CompoundVariable("comb_energy", variables=[testvar, testvar])
    assert (comp_var < testvar)
    assert isinstance(hash(comp_var), int)
    assert isinstance(comp_var.__repr__(), str)
    comp_var.harvest(prepare_testtable.realpath())
    assert comp_var.is_harvested

    list_var = v.VariableList("list_energy", variables=[testvar, testvar])
    assert isinstance(hash(list_var), int)
    assert isinstance(list_var.__repr__(), str)
    list_var.harvest(prepare_testtable.realpath())
    assert list_var.is_harvested
    assert len(list_var.data) == 2


def test_Cut():
    testcut = cut.Cut(*generate_test_cuts())
    assert sorted(testcut.variablenames) == sorted(generate_test_cut_variablenames())
    assert len([k for k in testcut]) == len(generate_test_cut_variablenames())
    assert isinstance(testcut.__repr__(), str)


def test_init_Cut_w_condition():
    testcut = cut.Cut(*generate_test_cuts(), name="Test", condition=np.ones(20))
    assert sorted(testcut.variablenames) == sorted(generate_test_cut_variablenames())
    assert len([k for k in testcut]) == len(generate_test_cut_variablenames())
    assert isinstance(testcut.__repr__(), str)


def test_simcat(prepare_testtable):
    import numpy as np
    import os.path

    sim = cat.Simulation("neutrino")
    filename = str(prepare_testtable.realpath())
    sim.get_files(os.path.split(filename)[0], ending=".h5", prefix="", sanitizer=lambda x : "test" in x)
    assert len(sim.files) > 0


    energy = v.Variable("energy",bins=np.linspace(0,10,20),\
                         label=r"$\log(E_{rec}/$GeV$)$",\
                         definitions=[("readout","energy")])
    #print (energy)
    sim.add_variable(energy)
    for var in [mc_p_en, mc_p_ty, mc_p_ze, mc_p_we, mc_p_gw, mc_p_ts]:
        sim.add_variable(var)

    sim.read_variables()
    assert sim.is_harvested
    assert sim.mc_p_readout

    assert len(sim.get("energy")) > 0
    assert len(sim) > 0
    assert "energy" in sim.variablenames
    def wf(x,y,**kwargs):
        return np.ones(TESTDATALEN)
    sim.set_weightfunction(wf)
    sim.get_weights(lambda x: x)
    #assert np.isfinite(sim.livetime)
    assert sim.livetime == 1.

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


def test_dataset(prepare_testtable):
    import os.path

    exp = cat.Data("exp")
    sim = cat.Simulation("nu")
    data = ds.Dataset(exp, sim)
    assert isinstance(data.__repr__(), str)
    assert len(data.categorynames) == 2
    filename = str(prepare_testtable.realpath())
    exp.get_files(os.path.split(filename)[0], ending=".h5", prefix="", sanitizer=lambda x : "test" in x)
    assert len(exp.files) > 0
    energy = v.Variable("energy",bins=np.linspace(0,10,20),\
                         label=r"$\log(E_{rec}/$GeV$)$",\
                         definitions=[("readout","energy")])
    #print (energy)
    exp.add_variable(energy)
    exp.read_variables()

    sim = cat.Simulation("neutrino")
    filename = str(prepare_testtable.realpath())
    sim.get_files(os.path.split(filename)[0], ending=".h5", prefix="", sanitizer=lambda x : "test" in x)
    assert len(sim.files) > 0
    for var in [mc_p_en, mc_p_ty, mc_p_ze, mc_p_we, mc_p_gw, mc_p_ts]:
        sim.add_variable(var)
    sim.read_mc_primary()
    assert sim.mc_p_readout
    assert len(sim) > 0

