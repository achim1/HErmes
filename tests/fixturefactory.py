import pytest

TESTDATALEN = 10000
SPARSETESTDATALEN = 7000


@pytest.fixture(scope='session')
def prepare_testtable(tmpdir_factory):
    from tables import IsDescription, Int64Col, Float64Col, open_file
    import numpy as np

    class TestParticle(IsDescription):
        energy = Float64Col()  # double (double-precision)
        x = Float64Col()

    class Header(IsDescription):
        start = Float64Col()
        stop = Float64Col()
        eventid = Int64Col()
        runid = Int64Col()

    class MCPrimary(IsDescription):
        energy = Float64Col()
        type = Float64Col()
        zenith = Float64Col()

    class CorsikaWeightMap(IsDescription):
        DiplopiaWeight = Float64Col()
        Weight = Float64Col()
        TimeScale = Float64Col()

    class FailedReco(IsDescription):
        energy = Float64Col()
        

    data = np.random.normal(5,10, TESTDATALEN)
    testdatafile = tmpdir_factory.mktemp('data').join("testdata.h5")
    testfile = open_file(str(testdatafile.realpath()), mode="w")
    #group = testfile.create_group("/", 'detector', 'Detector information')

    table = testfile.create_table("/", 'header', Header, "header example")
    header = table.row
    for i in range(TESTDATALEN):
        header['eventid'] = i
        header["runid"] = 100 + i
        header['start'] = float(1000.*i)
        header['stop'] = float(1000*i + 100)
        header.append()
    table.flush()

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
    table = testfile.create_table("/", 'FailedReco', FailedReco, "reco with different length")
    particle = table.row
    for i in range(TESTDATALEN -10):
        particle['energy'] = float(100. * data[i] ** 2)
    table.flush()
    table = testfile.create_table("/", 'CorsikaWeightMap', CorsikaWeightMap, "cwm example")
    particle = table.row
    for i in range(TESTDATALEN):
        particle['DiplopiaWeight'] = float(data[i])
        particle['Weight'] = float(data[i])
        particle['TimeScale'] = float(100. * data[i] ** 2)
        particle.append()
    table.flush()
    testfile.flush()
    testfile.close()
    return testdatafile

@pytest.fixture(scope='session')
def prepare_sparser_testtable(tmpdir_factory):
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

    class FailedReco(IsDescription):
        energy = Float64Col()
        

    data = np.random.normal(5,10, SPARSETESTDATALEN)
    testdatafile = tmpdir_factory.mktemp('data').join("sparse_testdata.h5")
    testfile = open_file(str(testdatafile.realpath()), mode="w")
    #group = testfile.create_group("/", 'detector', 'Detector information')
    table = testfile.create_table("/", 'readout', TestParticle, "Readout example")
    particle = table.row
    for i in range(SPARSETESTDATALEN):
        particle['x'] = float(data[i])
        particle['energy'] = float(100.*data[i]**2)
        particle.append()
    table.flush()
    table = testfile.create_table("/", 'MCPrimary', MCPrimary, "mc primary example")
    particle = table.row
    for i in range(SPARSETESTDATALEN):
        particle['type'] = float(data[i])
        particle['energy'] = float(100. * data[i] ** 2)
        particle['zenith'] = float(100. * data[i] ** 2)
        particle.append()
    table.flush()
    table = testfile.create_table("/", 'FailedReco', FailedReco, "reco with different length")
    particle = table.row
    for i in range(SPARSETESTDATALEN -10):
        particle['energy'] = float(100. * data[i] ** 2)
    table.flush()
    table = testfile.create_table("/", 'CorsikaWeightMap', CorsikaWeightMap, "cwm example")
    particle = table.row
    for i in range(SPARSETESTDATALEN):
        particle['DiplopiaWeight'] = float(data[i])
        particle['Weight'] = float(data[i])
        particle['TimeScale'] = float(100. * data[i] ** 2)
        particle.append()
    table.flush()
    testfile.close()
    return testdatafile

