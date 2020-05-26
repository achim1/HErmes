"""
Multi-step operations which might be ultimatly performed on variables in 
a dataset
"""

def construct_slices(name, bins):
    """
    Prepare a set of cuts for the variable with name "name" in the dataset
    This will just create the bins. This has then to be handed over
    to the HErmes.cut.Cut class for further application on a dataset.

    Args:
        name (str)    : The name of the variable in the dataset
        bins (array)  : bincenters of the slices

    Returns:
        tuple (list of strings, list of cuttuples)
    """
    l_binedges = bins[:-1]
    r_binedges = bins[1:]
    cuts = [[(name, ">", i), (name, "<=", j)] for i, j in zip(l_binedges, r_binedges)]
    labels = ["{:4.2f} - {:4.2f}".format(i, j) for i,j in zip(l_binedges, r_binedges)]
    return labels, cuts



