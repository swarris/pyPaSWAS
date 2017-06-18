/** Direction definitions for the direction matrix. These are needed for the trace back */
#define NO_DIRECTION ${NO_DIRECTION}
#define STOP_DIRECTION ${STOP_DIRECTION}

#define UPPER_LEFT_DIRECTION ${UP_LEFT_DIRECTION}
#define UPPER_DIRECTION ${UP_DIRECTION}
#define LEFT_DIRECTION ${LEFT_DIRECTION}

/** Direction definitions for affine gap penalty **/
#define MAIN_MATRIX ${UP_LEFT_DIRECTION}
#define I_MATRIX ${UP_DIRECTION}
#define J_MATRIX ${LEFT_DIRECTION}

#define A_DIRECTION (${UP_LEFT_DIRECTION} << 2)
#define B_DIRECTION (${UP_DIRECTION} <<2)
#define C_DIRECTION (${LEFT_DIRECTION} <<2)
#define DIRECTION_MASK 12
#define MATRIX_MASK 3

