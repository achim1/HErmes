"""
Miscellaneous tools
"""

__all__ = ["files","itools","logger"]

import resource

from logger import Logger

########################################################

def GetTiming(func):
    """
    Use as decorator to get the effective execution time of the decorated function

    Args:
        func (func): Measure the execution time of this function

    Returns:
        func: wrapped func
    """

    from time import time


    def wrapper(*args,**kwargs):
        t1 = time()
        res = func(*args,**kwargs)
        t2 = time()
        seconds = t2 -t1
        mins  = int(seconds)/60
        hours = int(seconds)/3600
        res_seconds  = int(seconds)%3600
        mins         = int(res_seconds)/60
        left_seconds = int(res_seconds)%60
        Logger.info('Execution of %s took %i hours, %i mins and %i seconds' %(func.func_name,hours,mins,left_seconds))
        Logger.info('Execution of %s took %f seconds' %(func.func_name,seconds))
        max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        Logger.info('Execution might have needed %i kB in memory (highly uncertain)!' %max_mem)
        return res

    return wrapper




