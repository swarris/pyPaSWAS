''' Custom CUDA exceptions '''


# TODO: pyCuda already has some predefined exceptions, are these still necessary?
class InvalidOptionException(Exception):
    '''
    This class is used to give a consistent user interface to exceptions that are specific for pyPaSWAS.
    '''


class HardwareException(Exception):
    '''
    This class is used to warn the user for unsupported hardware.
    '''


class CudaException(Exception):
    '''
    This class is used to warn the user that an exception in the cuda code has occurred.
    '''
