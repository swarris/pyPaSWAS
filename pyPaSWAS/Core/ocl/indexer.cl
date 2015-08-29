
#define INDEX_SIZE ${size}
#define BLOCK_SIZE ${block}
#define STEP_SIZE ${stepSize}

#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable
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
	
		s_distances[thread] = (float) index[block+threadPlus1] - (float)query[threadPlus1];
		s_distances[thread] *= s_distances[thread];
	
	
		unsigned int offset = 1;
	
		for (int i=INDEX_SIZE>>1; i > 0 ; i >>=1 ) {
			barrier(CLK_LOCAL_MEM_FENCE);
			if (thread < i) {
			    int ai = offset*(2*thread+1)-1;
			    int bi = offset*(2*thread+2)-1;
			    s_distances[bi] += s_distances[ai];
			}
			offset *= 2;
		}
		if (thread == 0){
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
