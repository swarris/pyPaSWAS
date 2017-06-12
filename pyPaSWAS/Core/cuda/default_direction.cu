/** Direction definitions for the direction matrix. These are needed for the trace back */
#define NO_DIRECTION ${NO_DIRECTION}
#define UPPER_LEFT_DIRECTION ${UP_LEFT_DIRECTION}
#define UPPER_DIRECTION ${UP_DIRECTION}
#define LEFT_DIRECTION ${LEFT_DIRECTION}
#define STOP_DIRECTION ${STOP_DIRECTION}

/** Direction definitions for affine gap penalty **/
#define NO_DIRECTION_I ${NO_DIRECTION}
#define UPPER_LEFT_DIRECTION_I (${UP_LEFT_DIRECTION} << 2)
#define UPPER_DIRECTION_I (${UP_DIRECTION} << 2)
#define LEFT_DIRECTION_I (${LEFT_DIRECTION} << 2)
#define STOP_DIRECTION_I ${STOP_DIRECTION}

#define NO_DIRECTION_J ${NO_DIRECTION}
#define UPPER_LEFT_DIRECTION_J (${UP_LEFT_DIRECTION} << 4)
#define UPPER_DIRECTION_J (${UP_DIRECTION} << 4)
#define LEFT_DIRECTION_J (${LEFT_DIRECTION} << 4)
#define STOP_DIRECTION_J ${STOP_DIRECTION}
