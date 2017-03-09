import pytest

import pyevsel.variables.categories as cat
import pyevsel.variables.variables as v

# test the magic keywords ..
from pyevsel.variables import magic_keywords

from pyevsel.variables import cut

def generate_test_cuts():
    return [("mc_p_energy", ">=", 5), ("charge","<=", 10),\
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

    data = np.random.normal(5,10, 10000)
    testdatafile = tmpdir_factory.mktemp('data').join("testdata.h5")
    testfile = open_file(str(testdatafile.realpath()), mode="w")
    #group = testfile.create_group("/", 'detector', 'Detector information')
    table = testfile.create_table("/", 'readout', TestParticle, "Readout example")
    particle = table.row
    for i in range(10000):
        particle['x'] = float(data[i])
        particle['energy'] = float(100.*data[i]**2)
        particle.append()
    table.flush()
    testfile.close()
    return testdatafile

def test_Cut():
    # FIXME! Test the condition
    testcut = cut.Cut(*generate_test_cuts())
    assert sorted(testcut.variablenames) == sorted(generate_test_cut_variablenames())
    assert len([k for k in testcut]) == len(generate_test_cut_variablenames())
    assert isinstance(testcut.__repr__(), str)


def test_init_Cut_w_condition():
    assert False, "Test missing!"

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
    sim.read_variables()
    assert len(sim.get("energy")) > 0

def test_datcat(prepare_testtable):
    import numpy as np
    import os.path

    sim = cat.Data("exp")
    filename = str(prepare_testtable.realpath())
    sim.get_files(os.path.split(filename)[0], ending=".h5", prefix="", sanitizer=lambda x : "test" in x)
    assert len(sim.files) > 0
    energy = v.Variable("energy",bins=np.linspace(0,10,20),\
                         label=r"$\log(E_{rec}/$GeV$)$",\
                         definitions=[("readout","energy")])
    #print (energy)
    sim.add_variable(energy)
    sim.read_variables()
    assert len(sim.get("energy")) > 0


