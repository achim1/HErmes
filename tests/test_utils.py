import pytest

from HErmes.utils import files

from fixturefactory import prepare_testtable
# helper functions generating test data
# provide a list of input/output


@pytest.fixture(params = [(10,0),\
                          (20,0),\
                          (30,0)])
def prepare_test_data_for_logger(request):
    return request.param

@pytest.fixture(params = [("test.i3.bz2",["test",".i3.bz2"]),\
                          ("foobar.hdf",["foobar",".hdf"]),
                          ("foobar.h5",["foobar",".h5"]),
                          ("file.root", ["file",".root"])])
def prepare_test_data_for_strip_all_endings(request):
    return request.param


def test_group_by_regex():
    names = [ "data_Run51000000.i3.bz2","test.GCD.i3.bz2",\
             "foo_Run51000000.i3.bz2", "foo_Run51000000.i3.bz2"]
    result = files.group_names_by_regex(names)
    assert len(result[51000000]) == 3


def test_check_hdf_integrity(prepare_testtable):

    files.check_hdf_integrity([str(prepare_testtable.realpath())] )
    files.check_hdf_integrity([str(prepare_testtable.realpath())], checkfor="energy")

