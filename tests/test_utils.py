import pytest

from pyevsel.utils import logger, files, itools, timeit

from fixturefactory import prepare_testtable
# helper functions generating test data
# provide a list of input/output

@pytest.fixture(params = [(([],0),[]),\
                          (([],1),[]),\
                          (([1,2],2),([1],[2])),\
                          (([1,2,3],2),([1,2],[3])),\
                          (([1,2,3],4), ([1],[2],[3],[])),\
                          (([1,2,3,4],2), ([1,2],[3,4]))])
def prepare_test_data_for_slicer(request):
    return request.param

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


# test itools    
def test_slicer(prepare_test_data_for_slicer):
    (func_args, desired_result) = prepare_test_data_for_slicer
    result = [r for r in itools.slicer(*func_args)]
    if len(result) == 1:
        result = result[0]
    elif len(result) > 1:
        result = tuple(result)
        
    assert result == desired_result

# test logger
def test_logger(prepare_test_data_for_logger):
    import logging

    loglevel, __ = prepare_test_data_for_logger
    assert isinstance(logger.get_logger(loglevel), logging.Logger)
    
# test files
def test_strip_all_endings(prepare_test_data_for_strip_all_endings):
    (func_arg, desired_result) = prepare_test_data_for_strip_all_endings
    assert files.strip_all_endings(func_arg) == desired_result

def test_group_by_regex():
    names = [ "data_Run51000000.i3.bz2","test.GCD.i3.bz2",\
             "foo_Run51000000.i3.bz2", "foo_Run51000000.i3.bz2"]
    result = files.group_names_by_regex(names)
    assert len(result[51000000]) == 3

def test_timit():
    @timeit 
    def sleeper():
        import time
        time.sleep(2)

    sleeper()

def test_check_hdf_integrity(prepare_testtable):

    files.check_hdf_integrity(str(prepare_testtable.realpath()))
    files.check_hdf_integrity(str(prepare_testtable.realpath()), checkfor="energy")


