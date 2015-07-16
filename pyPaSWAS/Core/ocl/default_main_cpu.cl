/** maximum X per block (used in dimensions for blocks and amount of shared memory */
#define SHARED_X ${SHARED_X}
/** maximum Y per block (used in dimensions for blocks and amount of shared memory */
#define SHARED_Y ${SHARED_Y}

/** kernel contains a for-loop in which the score is calculated. */
#define DIAGONAL SHARED_X + SHARED_Y

/** amount of blocks across the X axis */
#define XdivSHARED_X (X/SHARED_X)
/** amount of blocks across the Y axis */
#define YdivSHARED_Y (Y/SHARED_Y)

#define WORKLOAD_X ${WORKLOAD_X}
#define WORKLOAD_Y ${WORKLOAD_Y}
#define WORKGROUP_X (SHARED_X/WORKLOAD_X)
#define WORKGROUP_Y (SHARED_Y/WORKLOAD_Y)

/** start of the alphabet, so scoringsmatrix index can be calculated */
#define characterOffset 'A'
/** character used to fill the sequence if length < X */
#define FILL_CHARACTER 'x'
#define FILL_SCORE -1E10

/** this value is used to allocate enough memory to store the starting points */
#define MAXIMUM_NUMBER_STARTING_POINTS (NUMBER_SEQUENCES*NUMBER_TARGETS*1000)

/**** Other definitions ****/

/** bit mask to get the negative value of a float, or to keep it negative */
#define SIGN_BIT_MASK 0x80000000

/* Scorings matrix for each sequence alignment */
typedef struct {
	float value[X+1][Y+1];
} ScoringsMatrix;

/* Scorings matrix for entire application */
typedef struct {
	ScoringsMatrix metaMatrix[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalMatrix;

typedef struct {
	float value[WORKGROUP_X][WORKGROUP_Y];
}  BlockMaxima;

typedef struct {
	BlockMaxima blockMaxima[XdivSHARED_X][YdivSHARED_Y];
} AlignMaxima;

/* Maximum matrix for entire application */
typedef struct {
	AlignMaxima alignMaxima[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalMaxima;


typedef struct {
	unsigned char value[X][Y];
} Direction;

typedef struct {
	Direction direction[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalDirection;

typedef struct {
	unsigned int sequence;
	unsigned int target;
	unsigned int blockX;
	unsigned int blockY;
	unsigned int valueX;
	unsigned int valueY;
	float score;
	float maxScore;
	float posScore;
} StartingPoint;

typedef struct {
	StartingPoint startingPoint[MAXIMUM_NUMBER_STARTING_POINTS];
} StartingPoints;

typedef struct {
	int s[1];
} Semaphore;

typedef struct {
	Semaphore semaphore[X][Y];
} Semaphores;

typedef struct {
	Semaphores semaphores[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalSemaphores;

void getSemaphore(__global int * semaphore) {
   int occupied = atom_xchg(semaphore, 1);
   while(occupied > 0)
   {
     occupied = atom_xchg(semaphore, 1);
   }
}

void releaseSemaphore(__global int * semaphore)
{
   int prevVal = atom_xchg(semaphore, 0);
}

__kernel void calculateScore(
		__global GlobalMatrix *matrix, 
		unsigned int x, 
		unsigned int y, 
		unsigned int numberOfBlocks,
		__global char *sequences, 
		__global char *targets, 
		__global GlobalMaxima *globalMaxima, 
		__global GlobalDirection *globalDirection) {
		
		
	// calculate indices:
	//unsigned int yDIVnumSeq = (blockIdx.y/NUMBER_SEQUENCES);
	unsigned int blockx = x - get_group_id(1)/NUMBER_TARGETS;//0<=(get_group_id(1)/NUMBER_TARGETS)<numberOfBlocks
	unsigned int blocky = y + get_group_id(1)/NUMBER_TARGETS;//0<=(get_group_id(1)/NUMBER_TARGETS)<numberOfBlocks
	unsigned int tIDx = get_local_id(0);
	unsigned int tIDy = get_local_id(1);
	unsigned int bIDx = get_group_id(0);
	unsigned int bIDy = get_group_id(1)%NUMBER_TARGETS;///numberOfBlocks;
	
	float thread_max = 0.0;
	for (int i=0; i < WORKGROUP_X + WORKGROUP_Y; ++i) {
		if(i==tIDx+tIDy) {
			for(int j=0; j<WORKLOAD_X; j++) {
				
				unsigned int aIDx = tIDx*WORKLOAD_X + j + blockx * SHARED_X; //0<=alignmentIDx<X
				unsigned int aXM1 = aIDx;
				
				++aIDx; //1<=alignmentIDx<=X
								
				int seqIndex1 = tIDx * WORKLOAD_X + j + bIDx * X + blockx * SHARED_X;
				
				char charS1 = sequences[seqIndex1];
				
				/** Number of target characters a single work-item is responsible for **/
				for(int k=0; k<WORKLOAD_Y; k++) {
					
					unsigned char direction = NO_DIRECTION;
					int seqIndex2 = tIDy*WORKLOAD_Y + k + bIDy * Y + blocky * SHARED_Y;
					char charS2 = targets[seqIndex2];

					unsigned int aIDy = tIDy*WORKLOAD_Y + k + blocky * SHARED_Y; //0<=alignmentIDy<Y
					unsigned int aYM1 = aIDy;
					
					++aIDy; //1<=alignmentIDy<=Y
		
					float currentScore = 0.0;
					float ulS = 0.0;
					float lS = 0.0; 
					float uS = 0.0;
					float innerScore = 0.0;
					
					
					innerScore = charS1 == FILL_CHARACTER || charS2 == FILL_CHARACTER ? FILL_SCORE : scoringsMatrix[charS1-characterOffset][charS2-characterOffset];					
					ulS = (*matrix).metaMatrix[bIDx][bIDy].value[aXM1][aYM1] + innerScore;
					lS = (*matrix).metaMatrix[bIDx][bIDy].value[aXM1][aIDy] + gapScore;
					uS = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aYM1] + gapScore;
					
					if (currentScore < lS) { // score comes from left
						currentScore = lS;
						direction = LEFT_DIRECTION;
					}
					if (currentScore < uS) { // score comes from above
						currentScore = uS;
						direction = UPPER_DIRECTION;
					}
					if (currentScore < ulS) { // score comes from upper left
						currentScore = ulS;
						direction = UPPER_LEFT_DIRECTION;
					}
					
					(*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy] = currentScore;
					(*globalDirection).direction[bIDx][bIDy].value[aXM1][aYM1] = direction;
					thread_max = fmax(currentScore, thread_max);
					
				}
			}

		}
	
		if(i-1==tIDx+tIDy) { //got a thread_maximum
			if(i==1) {
				//get the maximum value of surrounding blocks
				float maxPrev = 0.0;
				if (blockx && blocky) {
					maxPrev = fmax(fmax(globalMaxima->alignMaxima[bIDx][bIDy].blockMaxima[blockx-1][blocky-1].value[WORKGROUP_X-1][WORKGROUP_Y-1], globalMaxima->alignMaxima[bIDx][bIDy].blockMaxima[blockx-1][blocky].value[WORKGROUP_X-1][WORKGROUP_Y-1]), globalMaxima->alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky-1].value[WORKGROUP_X-1][WORKGROUP_Y-1]);
				}
				else if (blockx) {
					maxPrev = globalMaxima->alignMaxima[bIDx][bIDy].blockMaxima[blockx-1][blocky].value[WORKGROUP_X-1][WORKGROUP_Y-1];
				}
				else if (blocky) {
					maxPrev = globalMaxima->alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky-1].value[WORKGROUP_X-1][WORKGROUP_Y-1];
				}
				
				(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[0][0] = fmax(maxPrev, thread_max);
			} else if(!tIDx && tIDy) {
				(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[0][tIDy] = fmax((*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[0][tIDy-1],thread_max);
			} else if(tIDx && !tIDy) {
				(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx][0] = fmax((*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx-1][0],thread_max);
			} else if(tIDx && tIDy) {
				(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx][tIDy] = fmax((*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx-1][tIDy], fmax(thread_max,(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx][tIDy-1]));
			}
			
		}
		
	
		barrier(CLK_LOCAL_MEM_FENCE);
		
	}
	
	if (tIDx==WORKGROUP_X-1 && tIDy==WORKGROUP_Y-1) {
		(*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx][tIDy] = fmax(thread_max, fmax((*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx-1][tIDy], (*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[blockx][blocky].value[tIDx][tIDy-1]));
	}
}

__kernel void traceback(
		__global GlobalMatrix *matrix, 
		unsigned int x, 
		unsigned int y, 
		unsigned int numberOfBlocks, 
		__global GlobalMaxima *globalMaxima,
		__global GlobalDirection *globalDirection,
		__global GlobalDirection *globalDirectionZeroCopy, 
		volatile __global unsigned int *indexIncrement,
		__global StartingPoints *startingPoints, 
		__global float *maxPossibleScore, 
		__global GlobalSemaphores *globalSemaphores) {

		unsigned int blockx = x - get_group_id(1)/NUMBER_TARGETS;//0<=(get_group_id(1)/NUMBER_TARGETS)<numberOfBlocks
		unsigned int blocky = y + get_group_id(1)/NUMBER_TARGETS;//0<=(get_group_id(1)/NUMBER_TARGETS)<numberOfBlocks
		unsigned int tIDx = get_local_id(0);
		unsigned int tIDy = get_local_id(1);
		unsigned int bIDx = get_group_id(0);
		unsigned int bIDy = get_group_id(1)%NUMBER_TARGETS;///numberOfBlocks;
	    
		float maximum = (*globalMaxima).alignMaxima[bIDx][bIDy].blockMaxima[XdivSHARED_X-1][YdivSHARED_Y-1].value[WORKGROUP_X-1][WORKGROUP_Y-1];
	
	
		if(maximum >= MINIMUM_SCORE) {
			//float mpScore = maxPossibleScore[bIDx+inBlock];
			float mpScore = maxPossibleScore[bIDy*NUMBER_SEQUENCES+bIDx];
			for(int i=WORKGROUP_X+WORKGROUP_Y-1; i>=0; --i) {
				if(i==tIDx+tIDy) {
					for(int j=WORKLOAD_X-1; j>=0; j--){
						unsigned int aIDx = tIDx*WORKLOAD_X + j + blockx * SHARED_X; //0<=alignmentIDx<X
						unsigned int aXM1 = aIDx;
						++aIDx; //1<=alignmentIDx<=X
						for(int k=WORKLOAD_Y-1; k>=0; k--) {
							unsigned int aIDy = tIDy*WORKLOAD_Y + k + blocky * SHARED_Y; //0<=alignmentIDy<Y
							unsigned int aYM1 = aIDy;
							++aIDy; //1<=alignmentIDy<=Y
							float score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy];
							unsigned char direction = (*globalDirection).direction[bIDx][bIDy].value[aXM1][aYM1];

							if (direction == UPPER_LEFT_DIRECTION && score >= LOWER_LIMIT_SCORE * maximum && score >= mpScore) {
								// found starting point!
								unsigned int index = atom_inc(&indexIncrement[0]);
								// now copy this to host:
								StartingPoint start;
				                start.sequence = bIDx;
				                start.target = bIDy;
				                start.blockX = blockx;
				                start.blockY = blocky;
				                start.valueX = tIDx*WORKLOAD_X + j;
				                start.valueY = tIDy*WORKLOAD_Y + k;
				                //start.valueX = tIDx;
				                //start.valueY = tIDy;
								start.score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy];
								start.maxScore = maximum;
								start.posScore = mpScore;
								startingPoints->startingPoint[index] = start;
								//Mark this value
#ifdef NVIDIA
								(*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(score);
#else
								(*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy] = as_float(SIGN_BIT_MASK | as_int(score));
#endif
							}
							score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy];

							if (score < 0 && direction == UPPER_LEFT_DIRECTION) {
								score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy-1];
								if (score == 0.0) {
									direction = STOP_DIRECTION;
									globalDirectionZeroCopy->direction[bIDx][bIDy].value[aXM1][aYM1] = direction;
								}
								else {
									getSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1-1][aYM1-1].s[0]));
#ifdef NVIDIA
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(score));
#else
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy-1] = as_float(SIGN_BIT_MASK | as_int(score));
#endif
									releaseSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1-1][aYM1-1].s[0]));
								}
							}
							if (score < 0 && direction == LEFT_DIRECTION) {
								score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy];
								if (score == 0.0) {
									direction = STOP_DIRECTION;
									globalDirectionZeroCopy->direction[bIDx][bIDy].value[aXM1][aYM1] = direction;
								}
								else {
									getSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1-1][aYM1].s[0]));
#ifdef NVIDIA
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(score));
#else
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx-1][aIDy] = as_float(SIGN_BIT_MASK | as_int(score));
#endif
									releaseSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1-1][aYM1].s[0]));
								}
							}
							if (score < 0 && direction == UPPER_DIRECTION) {
								score = (*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy-1];
								if (score == 0.0) {
									direction = STOP_DIRECTION;
									globalDirectionZeroCopy->direction[bIDx][bIDy].value[aXM1][aYM1] = direction;
								}
								else {
									getSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1][aYM1-1].s[0]));
#ifdef NVIDIA
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(score));
#else
									(*matrix).metaMatrix[bIDx][bIDy].value[aIDx][aIDy-1] = as_float(SIGN_BIT_MASK | as_int(score));
#endif
									releaseSemaphore(&((*globalSemaphores).semaphores[bIDx][bIDy].semaphore[aXM1][aYM1-1].s[0]));
								}
							}


						}
					}
					
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
}
