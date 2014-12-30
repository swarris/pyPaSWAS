'''
TODO Add a proper introduction of the package.
'''
from pkg_resources import resource_filename  # @UnresolvedImport  # pylint: disable=E0611

# S-W direction parameters
NO_DIRECTION = 0
UPPER_LEFT_DIRECTION = 1
UPPER_DIRECTION = 2
LEFT_DIRECTION = 3
STOP_DIRECTION = 4
IN_ALIGNMENT = 13

def read_file(filename):
    ''' Reads a file and returns its contents as a single string '''
    with open(filename) as contents:
        return ''.join(contents.readlines())
