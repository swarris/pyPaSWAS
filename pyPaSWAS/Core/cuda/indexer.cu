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

extern "C"
__global__ void setToZero(float *comps);

extern "C"
__global__ void calculateQgrams(char *sequence, unsigned int q, unsigned int length, float *comps, float windowLength, float step, float fraction, char nAs);


__global__ void calculateDistance(int *index, int *query, float *distances, unsigned int *validComps,
		unsigned int *seqs,
		unsigned int *indexIncrement, float scale, unsigned int numSeqs, unsigned int length, float sliceDistance){


	unsigned int seq = blockIdx.y / BLOCK_SIZE;
	unsigned int blockY = blockIdx.y % BLOCK_SIZE;
	//unsigned int indexSeq = seq * STEP_SIZE;
	unsigned int comp = blockIdx.x * BLOCK_SIZE + blockY;
	unsigned int block = (comp)*(INDEX_SIZE+1);
	unsigned int threadPlus1 = threadIdx.x+1;
	unsigned int thread = threadIdx.x;

	if (comp < length) {
		__shared__ float s_distances[INDEX_SIZE];
		__shared__ int s_notEmptyQ[INDEX_SIZE];
		unsigned int indexValue = index[block+threadPlus1];
		unsigned int queryValue = query[threadPlus1+(seq*(INDEX_SIZE+1))];

		s_distances[thread] = (float) indexValue - (float)queryValue;
		s_distances[thread] *= s_distances[thread];

		s_notEmptyQ[thread] = indexValue | queryValue;

		unsigned int offset = 1;

		for (int i=INDEX_SIZE>>1; i > 0 ; i >>=1 ) {
			__syncthreads();
			if (thread < i) {
				int ai = offset*(2*thread+1)-1;
				int bi = offset*(2*thread+2)-1;
				s_distances[bi] += s_distances[ai];
				s_notEmptyQ[bi] |= s_notEmptyQ[ai];
			}
			offset *= 2;
		}
		__syncthreads();
		if (thread == 0 && s_notEmptyQ[INDEX_SIZE-1]){
			s_distances[INDEX_SIZE-1] = sqrt(s_distances[INDEX_SIZE-1])/scale;
			if (s_distances[INDEX_SIZE-1] < sliceDistance){
				unsigned int indices = atomicAdd(indexIncrement, 1);
				distances[indices] =  s_distances[INDEX_SIZE-1];
				validComps[indices] = comp;
				seqs[indices] = seq;
			}
		}

	}
}

__global__ void setToZero(float *comps){
	unsigned int index = 1+threadIdx.x + (INDEX_SIZE+1) * (blockIdx.x*BLOCK_SIZE + blockIdx.y);
	comps[index] = 0.0;
}

__global__ void calculateQgrams(char *sequence, unsigned int q, unsigned int length, float *comps, float windowLength, float step, float fraction, char nAs) {
	unsigned int seqLocation = threadIdx.x + INDEX_SIZE * blockIdx.x;
	if (seqLocation < length - q) {
		int localQgram = 0;
		int bit = 1;
		//for (int i=q-1; i >= 0; i--) {
		for (int i=0; i < q; i++) {
			if (localQgram >= 0) {
				char character = sequence[seqLocation+i];
				if (character == 'N')
					character = nAs;
				switch (character) {
					case 'A' : break;
					case 'T' : localQgram+=1*bit; break;
					case 'C' : localQgram+=2*bit; break;
					case 'G' : localQgram+=3*bit; break;
					default : localQgram = -1;
				}
				bit *= q;
			}
		}
		if (localQgram >= 0) {
			localQgram++;
			unsigned int startWindow = (seqLocation-windowLength) < 0 ? 0 : (unsigned int)ceil((seqLocation - windowLength) / step);
			unsigned int endWindow = (unsigned int)floor(seqLocation/step) < STEP_SIZE? (unsigned int)floor(seqLocation/step) : STEP_SIZE-1;
			comps[(startWindow * (INDEX_SIZE+1))]= (float)windowLength;

			for (unsigned int i =startWindow; i < endWindow; i++){
				atomicAdd(&comps[i * (INDEX_SIZE+1)+localQgram], fraction);
			}

		}

	}

}


