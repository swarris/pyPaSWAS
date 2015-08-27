#include <math.h>
#include <builtin_types.h>

#define INDEX_SIZE ${size}
#define BLOCK_SIZE ${block}
#define STEP_SIZE ${stepSize}

extern "C"
__global__ void calculateDistance(int *index, int *query, float *distances, unsigned int *validComps,
		unsigned int *seqs,
		unsigned int *indexIncrement,
		float scale, unsigned int numSeqs, unsigned int length, float sliceDistance);

__global__ void calculateDistance(int *index, int *query, float *distances, unsigned int *validComps,
		unsigned int *seqs,
		unsigned int *indexIncrement, float scale, unsigned int numSeqs, unsigned int length, float sliceDistance){

	__shared__ float s_distances[INDEX_SIZE];

	unsigned int seq = blockIdx.y / BLOCK_SIZE;
	unsigned int blockY = blockIdx.y % BLOCK_SIZE;
	//unsigned int indexSeq = seq * STEP_SIZE;
	unsigned int comp = blockIdx.x * BLOCK_SIZE + blockY;
	unsigned int block = (comp)*(INDEX_SIZE+1);
	unsigned int threadPlus1 = threadIdx.x+1;
	unsigned int thread = threadIdx.x;
	if (comp < length) {
		s_distances[thread] = (float) index[block+threadPlus1] - (float)query[threadPlus1+(seq*(INDEX_SIZE+1))];
		s_distances[thread] *= s_distances[thread];


		unsigned int offset = 1;

		for (int i=INDEX_SIZE>>1; i > 0 ; i >>=1 ) {
			__syncthreads();
			if (thread < i) {
				int ai = offset*(2*thread+1)-1;
				int bi = offset*(2*thread+2)-1;
				s_distances[bi] += s_distances[ai];
			}
			offset *= 2;
		}
		if (thread == 0){
			s_distances[INDEX_SIZE-1] = sqrt(s_distances[INDEX_SIZE-1])/scale;
			if (s_distances[INDEX_SIZE-1] <= sliceDistance){
				unsigned int index = atomicAdd(indexIncrement, 1);
				distances[index] =  s_distances[INDEX_SIZE-1];
				validComps[index] = comp;
				seqs[index] = seq;
			}
		}

	}
}
