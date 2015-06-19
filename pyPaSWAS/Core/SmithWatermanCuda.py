from pyPaSWAS.Core.SmithWaterman import SmithWaterman
from pyPaSWAS.Core.PaSWAS import Cudacode

from pyPaSWAS.Core import STOP_DIRECTION, LEFT_DIRECTION, NO_DIRECTION, UPPER_DIRECTION, UPPER_LEFT_DIRECTION, IN_ALIGNMENT
from pyPaSWAS.Core.Exceptions import HardwareException, InvalidOptionException

import pycuda.driver as driver
from pycuda.compiler import SourceModule
import numpy
import math

class SmithWatermanCuda(SmithWaterman):
    '''
    classdocs
    '''


    def __init__(self, params):
        
        self.device = 0

        self.cudacode = Cudacode(self.logger)
        # Compiling part of the CUDA code in advance
        self.cudacode.set_shared_xy_code(self.shared_x, self.shared_y)
        self.cudacode.set_direction_code(NO_DIRECTION, UPPER_LEFT_DIRECTION,
                                         UPPER_DIRECTION, LEFT_DIRECTION,
                                         STOP_DIRECTION)
        
        # Reference to the GPU device
        self._set_device(self.settings.device_number)
        self.logger.debug('Going to initialize device... with number {0}'.format(self.device))
        self._initialize_device(self.device)
        
    def __del__(self):
        '''Destructor. Removes the current running context'''
        self.logger.debug('Destructing SmithWaterman.')
        if (driver.Context is not None):  #@UndefinedVariable @IgnorePep8
            driver.Context.pop()  #@UndefinedVariable @IgnorePep8
            
    def _set_device(self, device):
        '''Sets the device number'''
        try:
            self.device = int(device) if device else self.device
        except ValueError:
            raise InvalidOptionException('device should be an int but is {0}'.format(device))
        
    def _initialize_device(self, device_number):
        '''
        Initalizes the GPU device and verifies its computational abilities.
        @param device_number: int value representing the device to use
        '''
        self.logger.debug('Initializing device {0}'.format(device_number))
        try:
            driver.init()  #@UndefinedVariable @IgnorePep8
            self.device = driver.Device(device_number)  #@UndefinedVariable @IgnorePep8
            self.device = self.device.make_context(flags=driver.ctx_flags.MAP_HOST).get_device()  #@UndefinedVariable @IgnorePep8
        except Exception as exception:
            raise HardwareException('Failed to initialize device. '
                                    'The following exception occurred: {0}'.format(str(exception)))  #@UndefinedVariable @IgnorePep8
        compute = self.device.compute_capability()
        if not ((compute[0] == 1 and compute[1] >= 2) or (compute[0] >= 2)):
            raise HardwareException('Failed to initialize device: '
                                    'need compute capability 1.2 or newer!')  #@UndefinedVariable @IgnorePep8
