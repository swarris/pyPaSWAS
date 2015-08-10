#include <math.h>
#include <builtin_types.h>

#define INDEX_SIZE ${size}
#define BLOCK_SIZE ${block}


extern "C"
__global__ void calculateDistance(int *index, int *query, float *distances, float scale);

__global__ void calculateDistance(int *index, int *query, float *distances, float scale){

	__shared__ float s_distances[INDEX_SIZE];

	unsigned int block = (blockIdx.x * BLOCK_SIZE + blockIdx.y)*(INDEX_SIZE+1);
	unsigned int threadPlus1 = threadIdx.x+1;
	unsigned int thread = threadIdx.x;
	s_distances[thread] = (float) index[block+threadPlus1] - (float)query[threadPlus1];
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
		distances[blockIdx.x*BLOCK_SIZE+blockIdx.y] =  sqrt(s_distances[INDEX_SIZE-1])/scale;
	}
}
