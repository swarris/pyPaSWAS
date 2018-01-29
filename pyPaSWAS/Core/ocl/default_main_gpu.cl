/** maximum X per block (used in dimensions for blocks and amount of shared memory */
#define SHARED_X ${SHARED_X}
/** maximum Y per block (used in dimensions for blocks and amount of shared memory */
#define SHARED_Y ${SHARED_Y}

/** kernel contains a for-loop in which the score is calculated. */
#define DIAGONAL SHARED_X + SHARED_Y

/** character used to fill the sequence if length < X */
#define FILL_CHARACTER '\0'
#define FILL_SCORE -1E10f

/** Set init for affine gap matrices */
#define AFFINE_GAP_INIT -1E10f

/**** Other definitions ****/

/** bit mask to get the negative value of a float, or to keep it negative */
#define SIGN_BIT_MASK 0x80000000

/* Scorings matrix for each thread block */
typedef struct {
    float value[SHARED_X][SHARED_Y];
} Matrix;

/* Direction matrix for each thread block */
typedef struct {
    unsigned char value[SHARED_X][SHARED_Y];
} Direction;

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

__kernel
__attribute__((reqd_work_group_size(SHARED_X, SHARED_Y, 1)))
void calculateScore(
        const unsigned int numberOfSequences,
        const unsigned int numberOfTargets,
        const unsigned int xDivSHARED_X,
        const unsigned int yDivSHARED_Y,
        __global Matrix *matrix,
        const unsigned int x,
        const unsigned int y,
        const __global char *sequences,
        const __global char *targets,
        __global float *globalMaxima,
        __global Direction *globalDirection,
        const __global float *maxPossibleScore,
        __global unsigned int *isTracebackRequired) {

    /**
     * shared memory block for calculations. It requires
     * extra (+1 in both directions) space to hold
     * Neighboring cells
     */
    __local float s_matrix[SHARED_X+1][SHARED_Y+1];

    // calculate indices:
    unsigned int blockx = x - get_group_id(0);
    unsigned int blocky = y + get_group_id(0);
    unsigned int tIDx = get_local_id(0);
    unsigned int tIDy = get_local_id(1);
    unsigned int bIDx = get_group_id(1);
    unsigned int bIDy = get_group_id(2);
    unsigned char direction = NO_DIRECTION;

    // Move pointers to current target and sequence
    const unsigned int offset = (bIDx * numberOfTargets + bIDy) * (xDivSHARED_X * yDivSHARED_Y);
    matrix += offset;
    globalMaxima += offset;
    globalDirection += offset;

    // indices of the current characters in both sequences.
    int seqIndex1 = tIDx + (bIDx * xDivSHARED_X + blockx) * SHARED_X;
    int seqIndex2 = tIDy + (bIDy * yDivSHARED_Y + blocky) * SHARED_Y;

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
    __local char s_seq1[SHARED_X];
    __local char s_seq2[SHARED_Y];

    // copy sequence data to shared memory (shared is much faster than global)
    if (!tIDy)
        s_seq1[tIDx] = sequences[seqIndex1];
    if (!tIDx)
        s_seq2[tIDy] = targets[seqIndex2];

    barrier(CLK_LOCAL_MEM_FENCE);

    // initialize outer parts of the matrix:
    if (!tIDx) {
        s_matrix[0][tIDy+1] = blockx ? matrix[(blockx-1) * yDivSHARED_Y + blocky].value[SHARED_X-1][tIDy] : 0.0f;
    }
    if (!tIDy) {
        s_matrix[tIDx+1][0] = blocky ? matrix[blockx * yDivSHARED_Y + (blocky-1)].value[tIDx][SHARED_Y-1] : 0.0f;
    }
    if (!tIDx && !tIDy){
        s_matrix[0][0] = blockx && blocky ? matrix[(blockx-1) * yDivSHARED_Y + (blocky-1)].value[SHARED_X-1][SHARED_Y-1] : 0.0f;
    }

    // set inner score (aka sequence match/mismatch score):
    char charS1 = s_seq1[tIDx];
    char charS2 = s_seq2[tIDy];

    innerScore = charS1 == FILL_CHARACTER || charS2 == FILL_CHARACTER ? FILL_SCORE : scoringsMatrix[charS1-characterOffset][charS2-characterOffset];

    // transpose the index
    ++tIDx;
    ++tIDy;

    for (int i = 0; i < DIAGONAL - 1; ++i) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (i == tXM1 + tYM1) {
            // calculate only when there are two valid characters
            // this is necessary when the two sequences are not of equal length
            // this is the SW-scoring of the cell:

            ulS = s_matrix[tXM1][tYM1] + innerScore;
            lS = s_matrix[tXM1][tIDy] + gapScore;
            uS = s_matrix[tIDx][tYM1] + gapScore;

            currentScore = fmax(fmax(0.0f, ulS), fmax(lS, uS));
            s_matrix[tIDx][tIDy] = currentScore; // copy score to matrix
        }
    }

    if (currentScore == ulS) // score comes from upper left
        direction = UPPER_LEFT_DIRECTION;
    if (currentScore == uS) // score comes from above
        direction = UPPER_DIRECTION;
    if (currentScore == lS) // score comes from left
        direction = LEFT_DIRECTION;
    currentScore = innerScore == FILL_SCORE ? 0.0f : currentScore;

    // copy end score to the scorings matrix:
    matrix[blockx * yDivSHARED_Y + blocky].value[tXM1][tYM1] = currentScore;
    globalDirection[blockx * yDivSHARED_Y + blocky].value[tXM1][tYM1] = direction;

    // Find maximum score
    __local float s_maxima[SHARED_X * SHARED_Y];

    const unsigned int lid = get_local_id(1) * SHARED_X + get_local_id(0);
    float m = currentScore;
    s_maxima[lid] = m;

    for (int stride = SHARED_X * SHARED_Y / 2; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < stride) {
            m = fmax(m, s_maxima[lid + stride]);
            s_maxima[lid] = m;
        }
    }

    if (lid == 0) {
        /* the next block is to get the maximum value from surrounding blocks. */
        if (blockx) {
            m = fmax(m, globalMaxima[(blockx-1) * yDivSHARED_Y + blocky]);
        }
        if (blocky) {
            m = fmax(m, globalMaxima[blockx * yDivSHARED_Y + (blocky-1)]);
        }
        globalMaxima[blockx * yDivSHARED_Y + blocky] = m;

        if (blockx == xDivSHARED_X - 1 && blocky == yDivSHARED_Y - 1) {
            if (m >= MINIMUM_SCORE && m >= maxPossibleScore[bIDy * numberOfSequences + bIDx]) {
                *isTracebackRequired = 1;
            }
        }
    }
}

__kernel
__attribute__((reqd_work_group_size(SHARED_X, SHARED_Y, 1)))
void calculateScoreAffineGap(
        const unsigned int numberOfSequences,
        const unsigned int numberOfTargets,
        const unsigned int xDivSHARED_X,
        const unsigned int yDivSHARED_Y,
        __global Matrix *matrix,
        __global Matrix *matrix_i,
        __global Matrix *matrix_j,
        const unsigned int x,
        const unsigned int y,
        const __global char *sequences,
        const __global char *targets,
        __global float *globalMaxima,
        __global Direction *globalDirection,
        const __global float *maxPossibleScore,
        __global unsigned int *isTracebackRequired) {

    /**
     * shared memory block for calculations. It requires
     * extra (+1 in both directions) space to hold
     * Neighboring cells
     */
    __local float s_matrix[SHARED_X+1][SHARED_Y+1];
    __local float s_matrix_i[SHARED_X+1][SHARED_Y+1];
    __local float s_matrix_j[SHARED_X+1][SHARED_Y+1];

    // calculate indices:
    unsigned int blockx = x - get_group_id(0);
    unsigned int blocky = y + get_group_id(0);
    unsigned int tIDx = get_local_id(0);
    unsigned int tIDy = get_local_id(1);
    unsigned int bIDx = get_group_id(1);
    unsigned int bIDy = get_group_id(2);
    unsigned char direction = NO_DIRECTION;

    // Move pointers to current target and sequence
    const unsigned int offset = (bIDx * numberOfTargets + bIDy) * (xDivSHARED_X * yDivSHARED_Y);
    matrix += offset;
    matrix_i += offset;
    matrix_j += offset;
    globalMaxima += offset;
    globalDirection += offset;

    // indices of the current characters in both sequences.
    int seqIndex1 = tIDx + (bIDx * xDivSHARED_X + blockx) * SHARED_X;
    int seqIndex2 = tIDy + (bIDy * yDivSHARED_Y + blocky) * SHARED_Y;

    // local scorings variables:
    float currentScore, currentScore_m, currentScore_i, currentScore_j, m_M, m_I, m_J;
    float innerScore = 0.0f;
    /**
     * tXM1 and tYM1 are to store the current value of the thread Index. tIDx and tIDy are
     * both increased with 1 later on.
     */
    unsigned int tXM1 = tIDx;
    unsigned int tYM1 = tIDy;

    // shared location for the parts of the 2 sequences, for faster retrieval later on:
    __local char s_seq1[SHARED_X];
    __local char s_seq2[SHARED_Y];

    // copy sequence data to shared memory (shared is much faster than global)
    if (!tIDy)
        s_seq1[tIDx] = sequences[seqIndex1];
    if (!tIDx)
        s_seq2[tIDy] = targets[seqIndex2];

    barrier(CLK_LOCAL_MEM_FENCE);

    // initialize outer parts of the matrix:
    if (!tIDx) {
        s_matrix[0][tIDy+1] = blockx ? matrix[(blockx-1) * yDivSHARED_Y + blocky].value[SHARED_X-1][tIDy] : 0.0f;
        s_matrix_i[0][tIDy+1] = blockx ? matrix_i[(blockx-1) * yDivSHARED_Y + blocky].value[SHARED_X-1][tIDy] : AFFINE_GAP_INIT;
        s_matrix_j[0][tIDy+1] = blockx ? matrix_j[(blockx-1) * yDivSHARED_Y + blocky].value[SHARED_X-1][tIDy] : AFFINE_GAP_INIT;
    }
    if (!tIDy) {
        s_matrix[tIDx+1][0] = blocky ? matrix[blockx * yDivSHARED_Y + (blocky-1)].value[tIDx][SHARED_Y-1] : 0.0f;
        s_matrix_i[tIDx+1][0] = blocky ? matrix_i[blockx * yDivSHARED_Y + (blocky-1)].value[tIDx][SHARED_Y-1] : AFFINE_GAP_INIT;
        s_matrix_j[tIDx+1][0] = blocky ? matrix_j[blockx * yDivSHARED_Y + (blocky-1)].value[tIDx][SHARED_Y-1] : AFFINE_GAP_INIT;
    }
    if (!tIDx && !tIDy){
        s_matrix[0][0] = blockx && blocky ? matrix[(blockx-1) * yDivSHARED_Y + (blocky-1)].value[SHARED_X-1][SHARED_Y-1] : 0.0f;
        s_matrix_i[0][0] = blockx && blocky ? matrix_i[(blockx-1) * yDivSHARED_Y + (blocky-1)].value[SHARED_X-1][SHARED_Y-1] : AFFINE_GAP_INIT;
        s_matrix_j[0][0] = blockx && blocky ? matrix_j[(blockx-1) * yDivSHARED_Y + (blocky-1)].value[SHARED_X-1][SHARED_Y-1] : AFFINE_GAP_INIT;
    }

    // set inner score (aka sequence match/mismatch score):
    char charS1 = s_seq1[tIDx];
    char charS2 = s_seq2[tIDy];

    innerScore = charS1 == FILL_CHARACTER || charS2 == FILL_CHARACTER ? FILL_SCORE : scoringsMatrix[charS1-characterOffset][charS2-characterOffset];

    // transpose the index
    ++tIDx;
    ++tIDy;

    for (int i = 0; i < DIAGONAL - 1; ++i) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (i == tXM1 + tYM1) {
            m_M = s_matrix[tXM1][tYM1] + innerScore;
            m_I = s_matrix_i[tXM1][tYM1] + innerScore;
            m_J = s_matrix_j[tXM1][tYM1] + innerScore;
            currentScore_m = fmax(fmax(0.0f, m_M), fmax(m_I, m_J));
            s_matrix[tIDx][tIDy] = currentScore_m; // copy score to matrix

            // now do I matrix:
            m_M = gapScore + gapExtension + s_matrix[tIDx][tYM1]; // score comes from m matrix (match)
            m_I = gapExtension + s_matrix_i[tIDx][tYM1]; // score comes from I matrix (gap in x)
            currentScore_i = fmax(m_I, m_M);
            currentScore_i = currentScore_i < 0 ? AFFINE_GAP_INIT : currentScore_i;
            s_matrix_i[tIDx][tIDy] = currentScore_i; // copy score to matrix

            // now do J matrix:
            m_M = gapScore + gapExtension + s_matrix[tXM1][tIDy]; // score comes from m matrix (match)
            m_J = gapExtension + s_matrix_j[tXM1][tIDy]; // score comes from J matrix (gap in y)
            currentScore_j = fmax(m_J, m_M);
            currentScore_j = currentScore_j < 0 ? AFFINE_GAP_INIT : currentScore_j;
            s_matrix_j[tIDx][tIDy] = currentScore_j; // copy score to matrix
        }
    }
    currentScore = fmax(currentScore_m, fmax(currentScore_i, currentScore_j));
    if (currentScore > 0) {
        if (currentScore == currentScore_m) {// direction from main
            direction = A_DIRECTION | MAIN_MATRIX;
        }
        else if(currentScore == currentScore_i) {// direction from I
            direction = B_DIRECTION | I_MATRIX;
        }
        else if(currentScore == currentScore_j){ // direction from J
            direction = C_DIRECTION | J_MATRIX;
        }
    }

    // copy end score to the scorings matrix:
    matrix[blockx * yDivSHARED_Y + blocky].value[tXM1][tYM1] = currentScore_m;
    matrix_i[blockx * yDivSHARED_Y + blocky].value[tXM1][tYM1] = currentScore_i;
    matrix_j[blockx * yDivSHARED_Y + blocky].value[tXM1][tYM1] = currentScore_j;
    globalDirection[blockx * yDivSHARED_Y + blocky].value[tXM1][tYM1] = direction;

    // Find maximum score
    __local float s_maxima[SHARED_X * SHARED_Y];

    const unsigned int lid = get_local_id(1) * SHARED_X + get_local_id(0);
    float m = currentScore;
    s_maxima[lid] = m;

    for (int stride = SHARED_X * SHARED_Y / 2; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < stride) {
            m = fmax(m, s_maxima[lid + stride]);
            s_maxima[lid] = m;
        }
    }

    if (lid == 0) {
        /* the next block is to get the maximum value from surrounding blocks. */
        if (blockx) {
            m = fmax(m, globalMaxima[(blockx-1) * yDivSHARED_Y + blocky]);
        }
        if (blocky) {
            m = fmax(m, globalMaxima[blockx * yDivSHARED_Y + (blocky-1)]);
        }
        globalMaxima[blockx * yDivSHARED_Y + blocky] = m;

        if (blockx == xDivSHARED_X - 1 && blocky == yDivSHARED_Y - 1) {
            if (m >= MINIMUM_SCORE && m >= maxPossibleScore[bIDy * numberOfSequences + bIDx]) {
                *isTracebackRequired = 1;
            }
        }
    }
}


__kernel
__attribute__((reqd_work_group_size(SHARED_X, SHARED_Y, 1)))
void traceback(
        const unsigned int numberOfSequences,
        const unsigned int numberOfTargets,
        const unsigned int xDivSHARED_X,
        const unsigned int yDivSHARED_Y,
        __global Matrix *matrix,
        const unsigned int x,
        const unsigned int y,
        const __global float *globalMaxima,
        __global Direction *globalDirection,
        volatile __global unsigned int *indexIncrement,
        __global StartingPoint *startingPoints,
        const __global float *maxPossibleScore) {

    /**
     * shared memory block for calculations. It requires
     * extra (+1 in both directions) space to hold
     * Neighboring cells
     */
    __local float s_matrix[SHARED_X+1][SHARED_Y+1];
    /**
     * shared memory for storing the maximum value of this alignment.
     */
    __local float s_maxima[1];
    __local float s_maxPossibleScore[1];

    // calculate indices:
    unsigned int blockx = x - get_group_id(0);
    unsigned int blocky = y + get_group_id(0);
    unsigned int tIDx = get_local_id(0);
    unsigned int tIDy = get_local_id(1);
    unsigned int bIDx = get_group_id(1);
    unsigned int bIDy = get_group_id(2);

    // Move pointers to current target and sequence
    const unsigned int offset = (bIDx * numberOfTargets + bIDy) * (xDivSHARED_X * yDivSHARED_Y);
    matrix += offset;
    globalMaxima += offset;
    globalDirection += offset;

    __local bool s_needsProcessing;

    if (!tIDx && !tIDy) {
        s_maxima[0] = globalMaxima[(xDivSHARED_X-1) * yDivSHARED_Y + (yDivSHARED_Y-1)];
        s_maxPossibleScore[0] = maxPossibleScore[bIDy*numberOfSequences+bIDx];

        s_needsProcessing = false;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (s_maxima[0]>= MINIMUM_SCORE) { // if the maximum score is below threshold, there is nothing to do

        s_matrix[tIDx][tIDy] = matrix[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy];

        unsigned char direction = globalDirection[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy];

        const bool isStartCandidate = (direction == UPPER_LEFT_DIRECTION && s_matrix[tIDx][tIDy] >= LOWER_LIMIT_SCORE * s_maxima[0] && s_matrix[tIDx][tIDy] >= s_maxPossibleScore[0]);
        // Check if there are continuing alignments (from neighbouring blocks) or new starting points.
        // Otherwise there is nothing to do.
        if (s_matrix[tIDx][tIDy] < 0.0f || isStartCandidate)
            s_needsProcessing = true;

        // wait until all elements have been copied to the shared memory block
        /**** sync barrier ****/
        barrier(CLK_LOCAL_MEM_FENCE);

        if (!s_needsProcessing)
            return;

        for (int i=DIAGONAL-1; i >= 0; --i) {

            if (i == tIDx + tIDy) {
                if (isStartCandidate && s_matrix[tIDx][tIDy] > 0.0f) { // is not a part of another alignment
                    // found starting point!
                    // reserve index:
                    unsigned int index = atom_inc(&indexIncrement[0]);
                    StartingPoint start;
                    //__global StartingPoint *start = &(startingPoints->startingPoint[index]);
                    start.sequence = bIDx;
                    start.target = bIDy;
                    start.blockX = blockx;
                    start.blockY = blocky;
                    start.valueX = tIDx;
                    start.valueY = tIDy;
                    start.score = s_matrix[tIDx][tIDy];
                    start.maxScore = s_maxima[0];
                    start.posScore = s_maxPossibleScore[0];
                    startingPoints[index] = start;
                    // mark this value:
                    s_matrix[tIDx][tIDy] = as_float(SIGN_BIT_MASK | as_int(s_matrix[tIDx][tIDy]));
                }

                if (s_matrix[tIDx][tIDy] < 0) {
                    const int dx = direction == UPPER_DIRECTION ? 0 : -1;
                    const int dy = direction == LEFT_DIRECTION ? 0 : -1;
                    int prevx = tIDx + dx;
                    int prevy = tIDy + dy;
                    if (prevx >= 0 && prevy >= 0) {
                        const float value = s_matrix[prevx][prevy];
                        if (value == 0.0f) {
                            direction = STOP_DIRECTION;
                        } else {
                            s_matrix[prevx][prevy] = as_float(SIGN_BIT_MASK | as_int(value));
                        }
                    } else {
                        int prevBlockx = blockx;
                        int prevBlocky = blocky;
                        if (prevx < 0) {
                            prevBlockx += dx;
                            prevx = SHARED_X - 1;
                        }
                        if (prevy < 0) {
                            prevBlocky += dy;
                            prevy = SHARED_Y - 1;
                        }
                        if (prevBlockx >= 0 && prevBlocky >= 0) {
                            const float value = matrix[prevBlockx * yDivSHARED_Y + prevBlocky].value[prevx][prevy];
                            if (value == 0.0f) {
                                direction = STOP_DIRECTION;
                            } else {
                                matrix[prevBlockx * yDivSHARED_Y + prevBlocky].value[prevx][prevy] = as_float(SIGN_BIT_MASK | as_int(value));
                            }
                        }
                    }
                }
            }
            /**** sync barrier ****/
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // copy end score to the scorings matrix:
        if (s_matrix[tIDx][tIDy] < 0) {
            matrix[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy] = s_matrix[tIDx][tIDy];
            globalDirection[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy] = direction;
        }
    }
}


__kernel
__attribute__((reqd_work_group_size(SHARED_X, SHARED_Y, 1)))
void tracebackAffineGap(
        const unsigned int numberOfSequences,
        const unsigned int numberOfTargets,
        const unsigned int xDivSHARED_X,
        const unsigned int yDivSHARED_Y,
        __global Matrix *matrix,
        __global Matrix *matrix_i,
        __global Matrix *matrix_j,
        const unsigned int x,
        const unsigned int y,
        const __global float *globalMaxima,
        __global Direction *globalDirection,
        volatile __global unsigned int *indexIncrement,
        __global StartingPoint *startingPoints,
        const __global float *maxPossibleScore) {

    /**
     * shared memory block for calculations. It requires
     * extra (+1 in both directions) space to hold
     * Neighboring cells
     */
    __local float s_matrix[SHARED_X+1][SHARED_Y+1];
    __local float s_matrix_i[SHARED_X+1][SHARED_Y+1];
    __local float s_matrix_j[SHARED_X+1][SHARED_Y+1];
    /**
     * shared memory for storing the maximum value of this alignment.
     */
    __local float s_maxima[1];
    __local float s_maxPossibleScore[1];

    // calculate indices:
    unsigned int blockx = x - get_group_id(0);
    unsigned int blocky = y + get_group_id(0);
    unsigned int tIDx = get_local_id(0);
    unsigned int tIDy = get_local_id(1);
    unsigned int bIDx = get_group_id(1);
    unsigned int bIDy = get_group_id(2);

    // Move pointers to current target and sequence
    const unsigned int offset = (bIDx * numberOfTargets + bIDy) * (xDivSHARED_X * yDivSHARED_Y);
    matrix += offset;
    matrix_i += offset;
    matrix_j += offset;
    globalMaxima += offset;
    globalDirection += offset;

    float value = 0.0f;

    if (!tIDx && !tIDy) {
        s_maxima[0] = globalMaxima[(xDivSHARED_X-1) * yDivSHARED_Y + (yDivSHARED_Y-1)];
        s_maxPossibleScore[0] = maxPossibleScore[bIDy*numberOfSequences+bIDx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (s_maxima[0]>= MINIMUM_SCORE) { // if the maximum score is below threshold, there is nothing to do
        unsigned char direction = DIRECTION_MASK & globalDirection[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy];
        unsigned char matrix_source = MATRIX_MASK & globalDirection[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy];

        s_matrix[tIDx][tIDy] = matrix[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy];
        s_matrix_i[tIDx][tIDy] = matrix_i[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy];
        s_matrix_j[tIDx][tIDy] = matrix_j[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy];


        // wait until all elements have been copied to the shared memory block
        /**** sync barrier ****/
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i=DIAGONAL-1; i >= 0; --i) {

            if ((i == tIDx + tIDy) && matrix_source == MAIN_MATRIX && s_matrix[tIDx][tIDy] >= LOWER_LIMIT_SCORE * s_maxima[0] && s_matrix[tIDx][tIDy] >= s_maxPossibleScore[0]) {
                // found starting point!
                // reserve index:
                unsigned int index = atom_inc(&indexIncrement[0]);
                StartingPoint start;
                //__global StartingPoint *start = &(startingPoints[index]);
                start.sequence = bIDx;
                start.target = bIDy;
                start.blockX = blockx;
                start.blockY = blocky;
                start.valueX = tIDx;
                start.valueY = tIDy;
                start.score = s_matrix[tIDx][tIDy];
                start.maxScore = s_maxima[0];
                start.posScore = s_maxPossibleScore[0];
                startingPoints[index] = start;
                // mark this value:
                s_matrix[tIDx][tIDy] = as_float(SIGN_BIT_MASK | as_int(s_matrix[tIDx][tIDy]));

            }

            barrier(CLK_LOCAL_MEM_FENCE);

            if ((i == tIDx + tIDy) && (
                    (s_matrix[tIDx][tIDy] < 0 && matrix_source == MAIN_MATRIX) ||
                    (s_matrix_i[tIDx][tIDy] < 0 && s_matrix_i[tIDx][tIDy] > AFFINE_GAP_INIT && matrix_source == I_MATRIX) ||
                    (s_matrix_j[tIDx][tIDy] < 0 && s_matrix_j[tIDx][tIDy] > AFFINE_GAP_INIT && matrix_source == J_MATRIX)
                    )) {
                    // check which matrix to go to:
                    switch (direction) {
                    case A_DIRECTION : // M
                        if (tIDx && tIDy){
                            value = s_matrix[tIDx-1][tIDy-1];
                            if (value == 0.0f)
                                direction = STOP_DIRECTION;
                            else
                                s_matrix[tIDx-1][tIDy-1] = as_float(SIGN_BIT_MASK | as_int(value));
                        }
                        else if (!tIDx && tIDy && blockx) {
                            value = matrix[(blockx-1) * yDivSHARED_Y + blocky].value[SHARED_X-1][tIDy-1];
                            if (value == 0.0f)
                                direction = STOP_DIRECTION;
                            else
                                matrix[(blockx-1) * yDivSHARED_Y + blocky].value[SHARED_X-1][tIDy-1] = as_float(SIGN_BIT_MASK | as_int(value));
                        }
                        else if (!tIDx && !tIDy && blockx && blocky) {
                            value = matrix[(blockx-1) * yDivSHARED_Y + (blocky-1)].value[SHARED_X-1][SHARED_Y-1];
                            if (value == 0.0f)
                                direction = STOP_DIRECTION;
                            else
                                matrix[(blockx-1) * yDivSHARED_Y + (blocky-1)].value[SHARED_X-1][SHARED_Y-1] = as_float(SIGN_BIT_MASK | as_int(value));
                        }
                        else if (tIDx && !tIDy && blocky) {
                            value = matrix[blockx * yDivSHARED_Y + (blocky-1)].value[tIDx-1][SHARED_Y-1];
                            if (value == 0.0f)
                                direction = STOP_DIRECTION;
                            else
                                matrix[blockx * yDivSHARED_Y + (blocky-1)].value[tIDx-1][SHARED_Y-1] = as_float(SIGN_BIT_MASK | as_int(value));
                        }

//direction = tracebackStepLeftUp(blockx, blocky, s_matrix, matrix, direction);
                        break;
                    case B_DIRECTION : // I
                        if (!tIDy) {
                            if (blocky) {
                                value = matrix_i[blockx * yDivSHARED_Y + (blocky-1)].value[tIDx][SHARED_Y-1];
                                matrix_i[blockx * yDivSHARED_Y + (blocky-1)].value[tIDx][SHARED_Y-1] = as_float(SIGN_BIT_MASK | as_int(value));
                            }
                        }
                        else {
                            value = s_matrix_i[tIDx][tIDy-1];
                            s_matrix_i[tIDx][tIDy-1] = as_float(SIGN_BIT_MASK | as_int(value));
                        }

                        //direction = tracebackStepUp(blockx, blocky, s_matrix_i, matrix_i, direction);
                        break;
                    case C_DIRECTION : // J
                        if (!tIDx){
                            if (blockx) {
                                value = matrix_j[(blockx-1) * yDivSHARED_Y + blocky].value[SHARED_X-1][tIDy];
                                matrix_j[(blockx-1) * yDivSHARED_Y + blocky].value[SHARED_X-1][tIDy] = as_float(SIGN_BIT_MASK | as_int(value));
                            }
                        }
                        else {
                            value = s_matrix_j[tIDx-1][tIDy];
                            s_matrix_j[tIDx-1][tIDy] = as_float(SIGN_BIT_MASK | as_int(value));
                        }

                        //direction = tracebackStepLeft(blockx, blocky, s_matrix_j, matrix_j, direction);
                        break;
                    }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // copy end score to the scorings matrix:
        if (matrix_source == MAIN_MATRIX) {
            matrix[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy] = s_matrix[tIDx][tIDy];
            globalDirection[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy] = direction;
        }
        else if (matrix_source == I_MATRIX) {
            matrix_i[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy] = s_matrix_i[tIDx][tIDy];
            globalDirection[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy] = direction;
        }
        else if (matrix_source == J_MATRIX) {
            matrix_j[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy] = s_matrix_j[tIDx][tIDy];
            globalDirection[blockx * yDivSHARED_Y + blocky].value[tIDx][tIDy] = direction;
        }
        /**** sync barrier ****/
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
