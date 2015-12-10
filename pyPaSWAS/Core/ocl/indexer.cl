
#define INDEX_SIZE ${size}
#define BLOCK_SIZE ${block}
#define STEP_SIZE ${stepSize}

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

/* self.d_compAll, self.d_comp, 
   self.d_distances, 
   self.d_validComps, 
   self.d_seqs, 
   self.d_index_increment, 
   numpy.float32(self.compositionScale),
   numpy.int32(len(seqs)), 
   numpy.int32(len(keys)), 
   numpy.float32(self.sliceDistance)
   */
__kernel void calculateDistance(
		__global int *index, 
		__global int *query, 
		__global float *distances,
		__global unsigned int *validComps,
		__global unsigned int *seqs,
		volatile __global unsigned int *indexIncrement,  
		float scale,
		unsigned int numSeqs, 
		unsigned int length, 
		float sliceDistance){
	
	
	//unsigned int block = (get_group_id(0) * BLOCK_SIZE + get_group_id(1))*(INDEX_SIZE+1);
	//unsigned int threadPlus1 = get_local_id(0)+1;
	//unsigned int thread = get_local_id(0);
	
	unsigned int seq = get_group_id(1) / BLOCK_SIZE;
	unsigned int blockY = get_group_id(1) % BLOCK_SIZE;
	unsigned int comp = get_group_id(0) * BLOCK_SIZE + blockY;
	unsigned int block = (comp)*(INDEX_SIZE+1);
	unsigned int threadPlus1 = get_local_id(0)+1;
	unsigned int thread = get_local_id(0);
	
	if (comp < length) {		
		__local float s_distances[INDEX_SIZE];
		__local int s_notEmptyQ[INDEX_SIZE];
		unsigned int indexValue = index[block+threadPlus1];
		unsigned int queryValue = query[threadPlus1+(seq*(INDEX_SIZE+1))];

		s_distances[thread] = (float) indexValue - (float)queryValue;
		s_distances[thread] *= s_distances[thread];
		
		s_notEmptyQ[thread] = indexValue | queryValue;
	
	
		unsigned int offset = 1;
	
		for (int i=INDEX_SIZE>>1; i > 0 ; i >>=1 ) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (thread < i) {
			    int ai = offset*(2*thread+1)-1;
			    int bi = offset*(2*thread+2)-1;
			    s_distances[bi] += s_distances[ai];
			    s_notEmptyQ[bi] |= s_notEmptyQ[ai];
			}
			offset *= 2;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (thread == 0 && s_notEmptyQ[INDEX_SIZE-1]){
			//distances[get_group_id(0) * BLOCK_SIZE + get_group_id(1)] =  sqrt(s_distances[INDEX_SIZE-1])/scale;
			s_distances[INDEX_SIZE-1] = sqrt(s_distances[INDEX_SIZE-1])/scale;

			if (s_distances[INDEX_SIZE-1] < sliceDistance){
				unsigned int indices = atom_inc(&indexIncrement[0]);
				distances[indices] =  s_distances[INDEX_SIZE-1];
				validComps[indices] = comp;
				seqs[indices] = seq;
			}
		}
	}
}

__kernel void setToZero(__global int *comps){
	unsigned int index = 1+ get_local_id(0) + (INDEX_SIZE+1) * (get_group_id(0)*BLOCK_SIZE + get_group_id(1));
	comps[index] = 0;
}

__kernel void scaleComp(__global float *comps, __global int *comps_int, float fraction){
	unsigned int index = 1+ get_local_id(0) + (INDEX_SIZE+1) * (get_group_id(0)*BLOCK_SIZE + get_group_id(1));
	comps[index] = (float)comps_int[index] / fraction;
	if (!get_local_id(0)) { // copy windows
		index--;
		comps[index] = (float)comps_int[index];
	}
}


__kernel void calculateQgrams(__global char *sequence, unsigned int q, unsigned int length, volatile __global int *comps, float windowLength, float step, int fraction, char nAs) {
	unsigned int seqLocation = get_local_id(0) + INDEX_SIZE * get_group_id(0);
	
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
			comps[(startWindow * (INDEX_SIZE+1))]= (int)windowLength;
			

			for (unsigned int i =startWindow; i < endWindow; i++){
				atom_add(&comps[i * (INDEX_SIZE+1)+localQgram], fraction);
			}
		}

	}
	

}
