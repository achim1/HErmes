import pytest

# test the magic keywords ..
from pyevsel.variables import magic_keywords

from pyevsel.variables import cut


# helper functions generating test data
def generate_test_cuts():
    return [("mc_p_energy", ">=", 5), ("charge","<=", 10),\
            ("energy",">", 10), ("velocity", "<", 42 )]


def generate_test_cut_variablenames():
    return [k[0] for k in generate_test_cuts()]


def test_Cut():
    # FIXME! Test the condition
    testcut = cut.Cut(*generate_test_cuts())
    assert sorted(testcut.variablenames) == sorted(generate_test_cut_variablenames())
    assert len([k for k in testcut]) == len(generate_test_cut_variablenames())
    assert isinstance(testcut.__repr__(), str)


def test_init_Cut_w_condition():
    assert False, "Test missing!"

