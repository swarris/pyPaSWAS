/** Direction definitions for the direction matrix. These are needed for the trace back */
#define NO_DIRECTION ${NO_DIRECTION}
#define STOP_DIRECTION ${STOP_DIRECTION}

#define UPPER_LEFT_DIRECTION ${UP_LEFT_DIRECTION}
#define UPPER_DIRECTION ${UP_DIRECTION}
#define LEFT_DIRECTION ${LEFT_DIRECTION}

/** Direction definitions for affine gap penalty **/
#define MAIN_MAIN_DIRECTION ${UP_LEFT_DIRECTION}
#define MAIN_I_DIRECTION ${UP_DIRECTION}
#define MAIN_J_DIRECTION ${LEFT_DIRECTION}
#define MAIN_MASK 3

#define I_MAIN_DIRECTION (${UP_LEFT_DIRECTION} << 2)
#define I_I_DIRECTION (${UP_DIRECTION} <<2)
#define I_J_DIRECTION (${LEFT_DIRECTION} <<2)
#define I_MASK 12

#define J_MAIN_DIRECTION (${UP_LEFT_DIRECTION} << 2)
#define J_I_DIRECTION (${UP_DIRECTION} << 2)
#define J_J_DIRECTION (${LEFT_DIRECTION} <<2)
#define J_MASK 48
