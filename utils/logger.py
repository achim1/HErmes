"""
Convenience function to provide easy logging
"""

import logging

LOGFORMAT = '%(levelname)s:%(message)s:%(module)s:%(funcName)s:%(lineno)d'

alertstring = lambda x :  "\033[0;31m" + x + "\033[00m"

def GetRootLogger(loglevel):
    """
    Get the root logger from the logging
    module -> All logging will end up at 
    the same place
    """   
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    formatter = logging.Formatter(LOGFORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    try:
        logger.handlers = [logger.handlers[1]]
    except IndexError:
        pass

    return logger

Logger = GetRootLogger(20)

