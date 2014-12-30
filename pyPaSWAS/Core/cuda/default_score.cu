/** Specifies the value for which a traceback can be started. If the
 * value in the alignment matrix is larger than or equal to
 * LOWER_LIMIT_SCORE * maxValue the traceback is started at this point.
 * A lower value for LOWER_LIMIT_SCORE will give more aligments.
 */

#define ${SCORE_TYPE}
#define LOWER_LIMIT_SCORE ${LOWER_LIMIT}f
#define MINIMUM_SCORE ${MINIMUM_SCORE}f
#define LOWER_LIMIT_MAX_SCORE ${MAX_SCORE}f

/** Only report alignments with a given minimum score. A good setting is:
 * (length shortest seq)*(lowest positive score) - (number of allowed gaps/mismatches)*(lowest negative score)
 * For testing: keep it low to get many alignments back.
 * @todo: make this a config at runtime.
 */

/** score used for a gap */
#define gapScore ${GAP_SCORE}f
#define HIGHEST_SCORE ${HIGHEST_SCORE}f

__constant__ float scoringsMatrix[26][26] = ${MATRIX};