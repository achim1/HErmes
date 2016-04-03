"""
Tools for managing iterables
"""

from logger import Logger

def slicer(list_to_slice,slices):
    """
    *Generator* Slice a list in  individual slices

    Args:
        list_to_slice (list): a list of items
        slices (int): how many lists will be sliced of
    
    Returns:
        list: slices of the given list
    """

    #FIXME: there must be something in the std lib..
    # implementing this because I am flying anyway
    # right now and have nothing to do..

    if slices == 0:
        slices = 1 # prevent ZeroDivisionError
    maxslice = len(list_to_slice)/slices
    if (maxslice*slices) < len(list_to_slice) :
        maxslice += 1
    Logger.info("Sliced list in %i slices with a maximum slice index of %i" %(slices,maxslice))
    for index in range(0,slices):
        lower_bound = index*maxslice
        upper_bound = lower_bound + maxslice
        thisslice = list_to_slice[lower_bound:upper_bound]
        #if not thisslice: #Do not emit empty lists!
        yield thisslice

def log_progress(sequence, every=None, size=None):
    """
    Display a progressbar in an IPython notebook
    stolen from https://github.com/alexanderkuk/log-progress

    Args:
        sequence (iterable): log the progress of this sequence

    Keyword Args:
        every :
        size  :

    Yields:
        sequence
    """

    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = size / 200     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{index} / ?'.format(index=index)
                else:
                    progress.value = index
                    label.value = u'{index} / {size}'.format(
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = str(index or '?')

