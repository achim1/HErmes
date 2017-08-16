"""
Convenience function to provide easy logging
"""

from future.utils import with_metaclass
import logging
import inspect
import os
import sys

#LOGFORMAT = '%(levelname)s:%(message)s:%(module)s:%(funcName)s:%(lineno)d'
LOGFORMAT = '%(levelname)s:%(message)s'
LOGLEVEL  = 30
alertstring = lambda x :  "\033[0;31m" + x + "\033[00m"

def get_logger(loglevel):
    """
    Get the root logger from the logging
    module -> All logging will end up at 
    the same place
    """

    def exception_handler(exctype, value, tb):
        logger.critical("Uncaught exception", exc_info=(exctype, value, tb))




    logger = logging.getLogger()
    logger.setLevel(loglevel)
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    formatter = logging.Formatter(LOGFORMAT)
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(ch)
    #FIXME: this seems not to be necessary
    if len(logger.handlers) > 1:
        logger.handlers = [logger.handlers[1]]

    sys.excepthook = exception_handler
    return logger


class CustomLoggerType(type):

    @staticmethod
    def __getattr__(item):
        logger = get_logger(LOGLEVEL)
        if len(logger.handlers) > 1:
            logger.handlers = [logger.handlers[1]]

        def wrapper(args):

            #for i in 0,1,2,3:
            #    frame = inspect.getouterframes(inspect.currentframe())[i]
            #    filename = frame[1]
            #    funcname = frame[2]
            #    lineno = frame[3]
            #    #print (i,filename, funcname, lineno)
            #    test = "{},{},{},{}".format(i, filename, funcname, lineno)
            #    print (test)
            #    del frame
            calframe = inspect.getouterframes(inspect.currentframe())[1]
            mname = os.path.split(calframe[1])[1].replace(".py","")
            lineno = calframe[2]
            fname = calframe[3]
            #if "module" in fname:
            #    fname = "module"

            args += ":{}:{}:{}".format(mname, fname, lineno)
            return getattr(logger, item)(args)

        return wrapper


class Logger(with_metaclass(CustomLoggerType,object)):
    pass


