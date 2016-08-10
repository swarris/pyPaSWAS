#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define NUMBER_SEQUENCES ${N_SEQUENCES}
#define NUMBER_TARGETS ${N_TARGETS}
#define X ${X}
#define Y ${Y}

/** start of the alphabet, so scoringsmatrix index can be calculated */
#define characterOffset '${CHAR_OFFSET}'

