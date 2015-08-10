
#define INDEX_SIZE ${size}
#define BLOCK_SIZE ${block}


__kernel void calculateDistance(__global int *index, __global int *query, __global float *distances, float scale){

	__local float s_distances[INDEX_SIZE];

	unsigned int block = (get_group_id(0) * BLOCK_SIZE + get_group_id(1))*(INDEX_SIZE+1);
	unsigned int threadPlus1 = get_local_id(0)+1;
	unsigned int thread = get_local_id(0);
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
		distances[get_group_id(0) * BLOCK_SIZE + get_group_id(1)] =  sqrt(s_distances[INDEX_SIZE-1])/scale;
	}
}
