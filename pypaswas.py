#!/usr/bin/python
from pyPaSWAS.pypaswasall import Pypaswas
import logging

if __name__ == '__main__':

    try:
        ppw = Pypaswas()
        ppw.run()
    except Exception as exception:
        # Show complete exception when running in DEBUG
        if (hasattr(ppw.settings, 'loglevel') and
            getattr(logging, 'DEBUG') == ppw.logger.getEffectiveLevel()):
            ppw.logger.exception(str(exception))
        else:
            print 'Something unforeseen caused the program to terminate.\n' \
                  'The internal error message was: ', ','.join(exception.args)
