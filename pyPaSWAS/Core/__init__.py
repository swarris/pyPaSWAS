'''
TODO Add a proper introduction of the package.
'''
from pkg_resources import resource_filename  # @UnresolvedImport  # pylint: disable=E0611

# S-W direction parameters
UPPER_LEFT_DIRECTION = 1
UPPER_DIRECTION = 2
LEFT_DIRECTION = 3

NO_DIRECTION = 2**5
STOP_DIRECTION = 2**6
IN_ALIGNMENT = 2**5 + 2**6

def read_file(filename):
    ''' Reads a file and returns its contents as a single string '''
    with open(filename) as contents:
        return ''.join(contents.readlines())
