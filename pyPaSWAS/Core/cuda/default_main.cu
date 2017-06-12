/** maximum X per block (used in dimensions for blocks and amount of shared memory */
#define SHARED_X ${SHARED_X}
/** maximum Y per block (used in dimensions for blocks and amount of shared memory */
#define SHARED_Y ${SHARED_Y}

/** kernel contains a for-loop in which the score is calculated. */
#define DIAGONAL SHARED_X + SHARED_Y

/** amount of score elements in a single block */
#define blockSize (SHARED_X * SHARED_Y)

/** amount of blocks across the X axis */
#define XdivSHARED_X (X/SHARED_X)
/** amount of blocks across the Y axis */
#define YdivSHARED_Y (Y/SHARED_Y)

/** character used to fill the sequence if length < X */
#define FILL_CHARACTER '\0'
#define FILL_SCORE -1E10f

/** this value is used to allocate enough memory to store the starting points */
#define MAXIMUM_NUMBER_STARTING_POINTS (NUMBER_SEQUENCES*NUMBER_TARGETS*1000)

/**** Other definitions ****/

/** bit mask to get the negative value of a float, or to keep it negative */
#define SIGN_BIT_MASK 0x80000000
#define MAX_LINE_LENGTH 500
#define AFFINE_GAP_INIT -1E10f

/* Scorings matrix for each thread block */
typedef struct {
    float value[SHARED_X][SHARED_Y];
}  LocalMatrix;

/* Scorings matrix for each sequence alignment */
typedef struct {
    LocalMatrix matrix[XdivSHARED_X][YdivSHARED_Y];
} ScoringsMatrix;

/* Scorings matrix for entire application */
typedef struct {
    ScoringsMatrix metaMatrix[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalMatrix;

typedef struct {
    float value[XdivSHARED_X][YdivSHARED_Y];
} BlockMaxima;

typedef struct {
    BlockMaxima blockMaxima[NUMBER_SEQUENCES][NUMBER_TARGETS];
} GlobalMaxima;

typedef struct {
    unsigned char value[SHARED_X][SHARED_Y];
} LocalDirection;

typedef struct {
    LocalDirection localDirection[XdivSHARED_X][YdivSHARED_Y];
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
    float value[NUMBER_TARGETS];
} TargetMaxima;

typedef struct {
    float value[NUMBER_SEQUENCES];
} SequenceMaxima;

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <builtin_types.h>
#include <time.h>

/**
 * The calculateScore function checks the alignment per block. It calculates the score for each cell in
 * shared memory.
 * @matrix   The scorings matrix
 * @x        The start x block position in the alignment to be calculated
 * @y        The start y block position in the alignment to be calculated
 * @numberOfBlocks The amount of blocks within an alignment which can be calculated
 * @seq1     The upper sequence in the alignment
 * @seq2     The left sequence in the alignment
 */
extern "C"
__global__ void calculateScore(GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks, char *sequences, char *targets, GlobalMaxima *globalMaxima, GlobalDirection *globalDirection);

extern "C"
__global__ void calculateScoreAffineGap(GlobalMatrix *matrix, GlobalMatrix *matrix_i, GlobalMatrix *matrix_j, unsigned int x, unsigned int y, unsigned int numberOfBlocks, char *sequences, char *targets, GlobalMaxima *globalMaxima, GlobalDirection *globalDirection);

extern "C"
__global__ void traceback(GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks, GlobalMaxima *globalMaxima, GlobalDirection *globalDirection, GlobalDirection *globalDirectionZeroCopy, unsigned int *indexIncrement, StartingPoints *startingPoints, float *maxPossibleScore);

extern "C"
__global__ void tracebackAffinegap(GlobalMatrix *matrix, GlobalMatrix *matrix_i, GlobalMatrix *matrix_j, unsigned int x, unsigned int y, unsigned int numberOfBlocks, GlobalMaxima *globalMaxima, GlobalDirection *globalDirection, GlobalDirection *globalDirectionZeroCopy, unsigned int *indexIncrement, StartingPoints *startingPoints, float *maxPossibleScore);


/**
 * The calculateScore function checks the alignment per block. It calculates the score for each cell in
 * shared memory
 * @matrix   The scorings matrix
 * @x        The start x block position in the alignment to be calculated
 * @y        The start y block position in the alignment to be calculated
 * @numberOfBlocks The amount of blocks within an alignment which can be calculated
 * @seq1     The upper sequence in the alignment
 * @seq2     The left sequence in the alignment
 */
__global__ void calculateScore(
        GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks,
        char *sequences, char *targets,
        GlobalMaxima *globalMaxima,
        GlobalDirection *globalDirection
        ) {
    /**
     * shared memory block for calculations. It requires
     * extra (+1 in both directions) space to hold
     * Neighboring cells
     */
    __shared__ float s_matrix[SHARED_X+1][SHARED_Y+1];
    /**
     * shared memory block for storing the maximum value of each neighboring cell.
     * Careful: the s_maxima[SHARED_X][SHARED_Y] does not contain the maximum value
     * after the calculation loop! This value is determined at the end of this
     * function.
     */
    __shared__ float s_maxima[SHARED_X][SHARED_Y];

    // calculate indices:
    //unsigned int yDIVnumSeq = (blockIdx.y/NUMBER_SEQUENCES);
    unsigned int blockx = x - blockIdx.y/NUMBER_TARGETS;//yDIVnumSeq;
    unsigned int blocky = y + blockIdx.y/NUMBER_TARGETS;//yDIVnumSeq;
    unsigned int tIDx = threadIdx.x;
    unsigned int tIDy = threadIdx.y;
    unsigned int bIDx = blockIdx.x;
    unsigned int bIDy = blockIdx.y%NUMBER_TARGETS;///numberOfBlocks;
    unsigned char direction = NO_DIRECTION;


    // indices of the current characters in both sequences.
    int seqIndex1 = tIDx + bIDx * X + blockx * SHARED_X;
    int seqIndex2 = tIDy + bIDy * Y + blocky * SHARED_Y;


    /* the next block is to get the maximum value from surrounding blocks. This maximum values is compared to the
     * first element in the shared score matrix s_matrix.
     */
    float maxPrev = 0.0f;
    if (!tIDx && !tIDy) {
        if (blockx && blocky) {
            maxPrev = max(max(globalMaxima->blockMaxima[bIDx][bIDy].value[blockx-1][blocky-1], globalMaxima->blockMaxima[bIDx][bIDy].value[blockx-1][blocky]), globalMaxima->blockMaxima[bIDx][bIDy].value[blockx][blocky-1]);
        }
        else if (blockx) {
            maxPrev = globalMaxima->blockMaxima[bIDx][bIDy].value[blockx-1][blocky];
        }
        else if (blocky) {
            maxPrev = globalMaxima->blockMaxima[bIDx][bIDy].value[blockx][blocky-1];
        }
    }
    // local scorings variables:
    float currentScore, ulS, lS, uS;
    float innerScore = 0.0f;
    /**
     * tXM1 and tYM1 are to store the current value of the thread Index. tIDx and tIDy are
     * both increased with 1 later on.
     */
    unsigned int tXM1 = tIDx;
    unsigned int tYM1 = tIDy;

    // shared location for the parts of the 2 sequences, for faster retrieval later on:
    __shared__ char s_seq1[SHARED_X];
    __shared__ char s_seq2[SHARED_Y];

    // copy sequence data to shared memory (shared is much faster than global)
    if (!tIDy)
        s_seq1[tIDx] = sequences[seqIndex1];
    if (!tIDx)
        s_seq2[tIDy] = targets[seqIndex2];

    // set both matrices to zero
    s_matrix[tIDx][tIDy] = 0.0f;
    s_maxima[tIDx][tIDy] = 0.0f;

    if (tIDx == SHARED_X-1  && ! tIDy)
        s_matrix[SHARED_X][0] = 0.0f;
    if (tIDy == SHARED_Y-1  && ! tIDx)
        s_matrix[0][SHARED_Y] = 0.0f;

    /**** sync barrier ****/
    s_matrix[tIDx][tIDy] = 0.0f;
    __syncthreads();

    // initialize outer parts of the matrix:
    if (!tIDx || !tIDy) {
        if (tIDx == SHARED_X-1)
            s_matrix[tIDx+1][tIDy] = 0.0f;
        if (tIDy == SHARED_Y-1)
            s_matrix[tIDx][tIDy+1] = 0.0f;
        if (blockx && !tIDx) {
            s_matrix[0][tIDy+1] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy];
        }
        if (blocky && !tIDy) {
            s_matrix[tIDx+1][0] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1];
        }
        if (blockx && blocky && !tIDx && !tIDy){
            s_matrix[0][0] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1];
        }
    }
    // set inner score (aka sequence match/mismatch score):
    char charS1 = s_seq1[tIDx];
    char charS2 = s_seq2[tIDy];

    innerScore = charS1 == FILL_CHARACTER || charS2 == FILL_CHARACTER ? FILL_SCORE : scoringsMatrix[charS1-characterOffset][charS2-characterOffset];

    // transpose the index
    ++tIDx;
    ++tIDy;
    // set shared matrix to zero (starting point!)
    s_matrix[tIDx][tIDy] = 0.0f;


    // wait until all elements have been copied to the shared memory block
        /**** sync barrier ****/
    __syncthreads();

    currentScore = 0.0f;

    for (int i=0; i < DIAGONAL; ++i) {
        if (i == tXM1+ tYM1) {
            // calculate only when there are two valid characters
            // this is necessary when the two sequences are not of equal length
            // this is the SW-scoring of the cell:

          ulS = s_matrix[tXM1][tYM1] + innerScore;
          lS = s_matrix[tXM1][tIDy] + gapScore;
          uS = s_matrix[tIDx][tYM1] + gapScore;

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
            s_matrix[tIDx][tIDy] = innerScore == FILL_SCORE ? 0.0 : currentScore; // copy score to matrix
        }

        else if (i-1 == tXM1 + tYM1 ){
                // use this to find max
            if (i==1) {
                s_maxima[0][0] = max(maxPrev, currentScore);
            }
            else if (!tXM1 && tYM1) {
                s_maxima[0][tYM1] = max(s_maxima[0][tYM1-1], currentScore);
            }
            else if (!tYM1 && tXM1) {
                s_maxima[tXM1][0] = max(s_maxima[tXM1-1][0], currentScore);
            }
            else if (tXM1 && tYM1 ){
                s_maxima[tXM1][tYM1] = max(s_maxima[tXM1-1][tYM1], max(s_maxima[tXM1][tYM1-1], currentScore));
            }
        }
        // wait until all threads have calculated their new score
            /**** sync barrier ****/
        __syncthreads();
    }
    // copy end score to the scorings matrix:
    (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tXM1][tYM1] = s_matrix[tIDx][tIDy];
    (*globalDirection).direction[bIDx][bIDy].localDirection[blockx][blocky].value[tXM1][tYM1] = direction;

    if (tIDx==SHARED_X && tIDy==SHARED_Y)
        globalMaxima->blockMaxima[bIDx][bIDy].value[blockx][blocky] = max(currentScore, max(s_maxima[SHARED_X-2][SHARED_Y-1], s_maxima[SHARED_X-1][SHARED_Y-2]));

    // wait until all threads have copied their score:
        /**** sync barrier ****/
    __syncthreads();
}

/**
 * The calculateScoreAffineGape function checks the alignment per block. It calculates the score for each cell in
 * shared memory using the affine gap penalty score
 * @matrix   The scorings matrix
 * @matrix_i The scorings matrix, gaps in X
 * @matrix_j The scorings matrix, gaps in Y
 * @x        The start x block position in the alignment to be calculated
 * @y        The start y block position in the alignment to be calculated
 * @numberOfBlocks The amount of blocks within an alignment which can be calculated
 * @seq1     The upper sequence in the alignment
 * @seq2     The left sequence in the alignment
 */
__global__ void calculateScoreAffineGap(GlobalMatrix *matrix, GlobalMatrix *matrix_i, GlobalMatrix *matrix_j,
		unsigned int x, unsigned int y, unsigned int numberOfBlocks,
        char *sequences, char *targets,
        GlobalMaxima *globalMaxima,
        GlobalDirection *globalDirection
        ) {
    /**
     * shared memory block for calculations. It requires
     * extra (+1 in both directions) space to hold
     * Neighboring cells
     */
    __shared__ float s_matrix[SHARED_X+1][SHARED_Y+1];
    __shared__ float s_matrix_i[SHARED_X+1][SHARED_Y+1];
    __shared__ float s_matrix_j[SHARED_X+1][SHARED_Y+1];
    /**
     * shared memory block for storing the maximum value of each neighboring cell.
     * Careful: the s_maxima[SHARED_X][SHARED_Y] does not contain the maximum value
     * after the calculation loop! This value is determined at the end of this
     * function.
     */
    __shared__ float s_maxima[SHARED_X][SHARED_Y];

    // calculate indices:
    //unsigned int yDIVnumSeq = (blockIdx.y/NUMBER_SEQUENCES);
    unsigned int blockx = x - blockIdx.y/NUMBER_TARGETS;//yDIVnumSeq;
    unsigned int blocky = y + blockIdx.y/NUMBER_TARGETS;//yDIVnumSeq;
    unsigned int tIDx = threadIdx.x;
    unsigned int tIDy = threadIdx.y;
    unsigned int bIDx = blockIdx.x;
    unsigned int bIDy = blockIdx.y%NUMBER_TARGETS;///numberOfBlocks;
    unsigned char direction = NO_DIRECTION;


    // indices of the current characters in both sequences.
    int seqIndex1 = tIDx + bIDx * X + blockx * SHARED_X;
    int seqIndex2 = tIDy + bIDy * Y + blocky * SHARED_Y;


    /* the next block is to get the maximum value from surrounding blocks. This maximum values is compared to the
     * first element in the shared score matrix s_matrix.
     */
    float maxPrev = 0.0f;
    if (!tIDx && !tIDy) {
        if (blockx && blocky) {
            maxPrev = max(max(globalMaxima->blockMaxima[bIDx][bIDy].value[blockx-1][blocky-1], globalMaxima->blockMaxima[bIDx][bIDy].value[blockx-1][blocky]), globalMaxima->blockMaxima[bIDx][bIDy].value[blockx][blocky-1]);
        }
        else if (blockx) {
            maxPrev = globalMaxima->blockMaxima[bIDx][bIDy].value[blockx-1][blocky];
        }
        else if (blocky) {
            maxPrev = globalMaxima->blockMaxima[bIDx][bIDy].value[blockx][blocky-1];
        }
    }
    // local scorings variables:
    float currentScore, ulS, lS, uS;
    float innerScore = 0.0f;
    /**
     * tXM1 and tYM1 are to store the current value of the thread Index. tIDx and tIDy are
     * both increased with 1 later on.
     */
    unsigned int tXM1 = tIDx;
    unsigned int tYM1 = tIDy;

    // shared location for the parts of the 2 sequences, for faster retrieval later on:
    __shared__ char s_seq1[SHARED_X];
    __shared__ char s_seq2[SHARED_Y];

    // copy sequence data to shared memory (shared is much faster than global)
    if (!tIDy)
        s_seq1[tIDx] = sequences[seqIndex1];
    if (!tIDx)
        s_seq2[tIDy] = targets[seqIndex2];

    // set matrices to zero
    s_matrix[tIDx][tIDy] = 0.0f;
    s_matrix_i[tIDx][tIDy] = AFFINE_GAP_INIT;
    s_matrix_j[tIDx][tIDy] = AFFINE_GAP_INIT;
    s_maxima[tIDx][tIDy] = 0.0f;

    if (tIDx == SHARED_X-1  && ! tIDy) {
        s_matrix[SHARED_X][0] = 0.0f;
        s_matrix_i[SHARED_X][0] = AFFINE_GAP_INIT;
        s_matrix_j[SHARED_X][0] = AFFINE_GAP_INIT;
    }
    if (tIDy == SHARED_Y-1  && ! tIDx) {
        s_matrix[0][SHARED_Y] = 0.0f;
        s_matrix_i[0][SHARED_Y] = AFFINE_GAP_INIT;
        s_matrix_j[0][SHARED_Y] = AFFINE_GAP_INIT;
    }

    /**** sync barrier ****/
    __syncthreads();

    // initialize outer parts of the matrix:
    if (!tIDx || !tIDy) {
        if (tIDx == SHARED_X-1) {
            s_matrix[tIDx+1][tIDy] = 0.0f;
            s_matrix_i[tIDx+1][tIDy] = AFFINE_GAP_INIT;
            s_matrix_j[tIDx+1][tIDy] = AFFINE_GAP_INIT;
        }
        if (tIDy == SHARED_Y-1) {
            s_matrix[tIDx][tIDy+1] = 0.0f;
            s_matrix_i[tIDx][tIDy+1] = AFFINE_GAP_INIT;
            s_matrix_j[tIDx][tIDy+1] = AFFINE_GAP_INIT;
        }
        if (blockx && !tIDx) {
            s_matrix[0][tIDy+1] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy];
            s_matrix_i[0][tIDy+1] = (*matrix_i).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy];
            s_matrix_j[0][tIDy+1] = (*matrix_j).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy];
        }
        if (blocky && !tIDy) {
            s_matrix[tIDx+1][0] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1];
            s_matrix_i[tIDx+1][0] = (*matrix_i).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1];
            s_matrix_j[tIDx+1][0] = (*matrix_j).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1];
        }
        if (blockx && blocky && !tIDx && !tIDy){
            s_matrix[0][0] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1];
            s_matrix_i[0][0] = (*matrix_i).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1];
            s_matrix_j[0][0] = (*matrix_j).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1];
        }
    }
    // set inner score (aka sequence match/mismatch score):
    char charS1 = s_seq1[tIDx];
    char charS2 = s_seq2[tIDy];

    innerScore = charS1 == FILL_CHARACTER || charS2 == FILL_CHARACTER ? FILL_SCORE : scoringsMatrix[charS1-characterOffset][charS2-characterOffset];

    // transpose the index
    ++tIDx;
    ++tIDy;
    // set shared matrix to zero (starting point!)
    s_matrix[tIDx][tIDy] = 0.0f;
    s_matrix_i[tIDx][tIDy] = AFFINE_GAP_INIT;
    s_matrix_j[tIDx][tIDy] = AFFINE_GAP_INIT;


    // wait until all elements have been copied to the shared memory block
        /**** sync barrier ****/
    __syncthreads();

    currentScore = 0.0f;

    for (int i=0; i < DIAGONAL; ++i) {
        if (i == tXM1+ tYM1) {
            // calculate only when there are two valid characters
            // this is necessary when the two sequences are not of equal length
            // this is the SW-scoring of the cell:

          ulS = s_matrix[tXM1][tYM1] + innerScore;
          lS = s_matrix[tXM1][tIDy] + gapScore;
          uS = s_matrix[tIDx][tYM1] + gapScore;

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
            s_matrix[tIDx][tIDy] = innerScore == FILL_SCORE ? 0.0 : currentScore; // copy score to matrix
        }

        else if (i-1 == tXM1 + tYM1 ){
                // use this to find max
            if (i==1) {
                s_maxima[0][0] = max(maxPrev, currentScore);
            }
            else if (!tXM1 && tYM1) {
                s_maxima[0][tYM1] = max(s_maxima[0][tYM1-1], currentScore);
            }
            else if (!tYM1 && tXM1) {
                s_maxima[tXM1][0] = max(s_maxima[tXM1-1][0], currentScore);
            }
            else if (tXM1 && tYM1 ){
                s_maxima[tXM1][tYM1] = max(s_maxima[tXM1-1][tYM1], max(s_maxima[tXM1][tYM1-1], currentScore));
            }
        }
        // wait until all threads have calculated their new score
            /**** sync barrier ****/
        __syncthreads();
    }
    // copy end score to the scorings matrix:
    (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tXM1][tYM1] = s_matrix[tIDx][tIDy];
    (*matrix_i).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tXM1][tYM1] = s_matrix_i[tIDx][tIDy];
    (*matrix_j).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tXM1][tYM1] = s_matrix_j[tIDx][tIDy];
    (*globalDirection).direction[bIDx][bIDy].localDirection[blockx][blocky].value[tXM1][tYM1] = direction;

    if (tIDx==SHARED_X && tIDy==SHARED_Y)
        globalMaxima->blockMaxima[bIDx][bIDy].value[blockx][blocky] = max(currentScore, max(s_maxima[SHARED_X-2][SHARED_Y-1], s_maxima[SHARED_X-1][SHARED_Y-2]));

    // wait until all threads have copied their score:
        /**** sync barrier ****/
    __syncthreads();
}




__global__ void traceback(GlobalMatrix *matrix, unsigned int x, unsigned int y, unsigned int numberOfBlocks, GlobalMaxima *globalMaxima, GlobalDirection *globalDirection, GlobalDirection *globalDirectionZeroCopy, unsigned int *indexIncrement, StartingPoints *startingPoints, float *maxPossibleScore) {
    /**
     * shared memory block for calculations. It requires
     * extra (+1 in both directions) space to hold
     * Neighboring cells
     */
    __shared__ float s_matrix[SHARED_X+1][SHARED_Y+1];
    /**
     * shared memory for storing the maximum value of this alignment.
     */
    __shared__ float s_maxima[1];
    __shared__ float s_maxPossibleScore[1];

    // calculate indices:
    unsigned int yDIVnumSeq = (blockIdx.y/NUMBER_TARGETS);
    unsigned int blockx = x - yDIVnumSeq;
    unsigned int blocky = y + yDIVnumSeq;
    unsigned int tIDx = threadIdx.x;
    unsigned int tIDy = threadIdx.y;
    unsigned int bIDx = blockIdx.x;
    unsigned int bIDy = blockIdx.y%NUMBER_TARGETS;

    float value;

    if (!tIDx && !tIDy) {
        s_maxima[0] = globalMaxima->blockMaxima[bIDx][bIDy].value[XdivSHARED_X-1][YdivSHARED_Y-1];
        s_maxPossibleScore[0] =  maxPossibleScore[bIDy*NUMBER_SEQUENCES+bIDx];
    }

    __syncthreads();
    if (s_maxima[0]>= MINIMUM_SCORE) { // if the maximum score is below threshold, there is nothing to do

        s_matrix[tIDx][tIDy] = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tIDx][tIDy];

        unsigned char direction = globalDirection->direction[bIDx][bIDy].localDirection[blockx][blocky].value[tIDx][tIDy];


        // wait until all elements have been copied to the shared memory block
        /**** sync barrier ****/
        __syncthreads();

        for (int i=DIAGONAL-1; i >= 0; --i) {

            if ((i == tIDx + tIDy) && direction == UPPER_LEFT_DIRECTION && s_matrix[tIDx][tIDy] >= LOWER_LIMIT_SCORE * s_maxima[0] && s_matrix[tIDx][tIDy] >= s_maxPossibleScore[0]) {
                // found starting point!
                // reserve index:
                unsigned int index = atomicAdd(indexIncrement, 1);
                // now copy this to host:
                StartingPoint *start = &(startingPoints->startingPoint[index]);
                start->sequence = bIDx;
                start->target = bIDy;
                start->blockX = blockx;
                start->blockY = blocky;
                start->valueX = tIDx;
                start->valueY = tIDy;
                start->score = s_matrix[tIDx][tIDy];
                start->maxScore = s_maxima[0];
                start->posScore = s_maxPossibleScore[0];
                //                startingPoints->startingPoint[index] = start;
                // mark this value:
                s_matrix[tIDx][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(s_matrix[tIDx][tIDy]));
            }
                
            __syncthreads();

            if ((i == tIDx + tIDy) && s_matrix[tIDx][tIDy] < 0 && direction == UPPER_LEFT_DIRECTION) {
                if (tIDx && tIDy){
                    value = s_matrix[tIDx-1][tIDy-1];
                    if (value == 0.0f)
                        direction = STOP_DIRECTION;
                    else
                        s_matrix[tIDx-1][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
                }
                else if (!tIDx && tIDy && blockx) {
                    value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy-1];
                    if (value == 0.0f)
                        direction = STOP_DIRECTION;
                    else
                        (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
                }
                else if (!tIDx && !tIDy && blockx && blocky) {
                    value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1];
                    if (value == 0.0f)
                        direction = STOP_DIRECTION;
                    else
                        (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky-1].value[SHARED_X-1][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
                }
                else if (tIDx && !tIDy && blocky) {
                    value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx-1][SHARED_Y-1];
                    if (value == 0.0f)
                        direction = STOP_DIRECTION;
                    else
                        (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx-1][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
                }
            }
            __syncthreads();

            if ((i == tIDx + tIDy) && s_matrix[tIDx][tIDy] < 0 && direction == UPPER_DIRECTION) {
                if (!tIDy) {
                    if (blocky) {
                        value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1];
                        if (value == 0.0f)
                            direction = STOP_DIRECTION;
                        else
                            (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky-1].value[tIDx][SHARED_Y-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
                    }
                }
                else {
                    value = s_matrix[tIDx][tIDy-1];
                    if (value == 0.0f)
                        direction = STOP_DIRECTION;
                    else
                        s_matrix[tIDx][tIDy-1] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
                }
            }

            __syncthreads();
            if ((i == tIDx + tIDy) && s_matrix[tIDx][tIDy] < 0 && direction == LEFT_DIRECTION) {
                if (!tIDx){
                    if (blockx) {
                        value = (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy];
                        if (value == 0.0f)
                            direction = STOP_DIRECTION;
                        else
                            (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx-1][blocky].value[SHARED_X-1][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
                    }
                }
                else {
                    value = s_matrix[tIDx-1][tIDy];
                    if (value == 0.0f)
                        direction = STOP_DIRECTION;
                    else
                        s_matrix[tIDx-1][tIDy] = __int_as_float(SIGN_BIT_MASK | __float_as_int(value));
                }
            }

            __syncthreads();

        }

        // copy end score to the scorings matrix:
        if (s_matrix[tIDx][tIDy] < 0) {
            (*matrix).metaMatrix[bIDx][bIDy].matrix[blockx][blocky].value[tIDx][tIDy] = s_matrix[tIDx][tIDy];
            globalDirectionZeroCopy->direction[bIDx][bIDy].localDirection[blockx][blocky].value[tIDx][tIDy] = direction;
        }
        /**** sync barrier ****/
        __syncthreads();
    }
}
