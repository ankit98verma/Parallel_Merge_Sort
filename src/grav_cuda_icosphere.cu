/*
 * CUDA blur
 */
#ifndef _GRAV_CUDA_ICOSPHERE_CU_
	#define _GRAV_CUDA_ICOSPHERE_CU_
#endif

#include <math.h>

#include "grav_cuda.cuh"


/* Local variables */

int ind2_faces;					// stores the index of the pointers_faces which points to the most updated faces array
triangle * pointers_faces[2];	// the pointer to contain the address of the faces array
triangle * dev_faces;			// contains the faces array in the device memory (GPU memory)
triangle * dev_faces_cpy;		// a copy for faces array  in the device memory (GPU memory)

int * pointers_inds[2];         // stores the index of the pointers_inds which points to the most updated indices array
int ind2_inds;					// the pointer to contain the address of the indices array
int * dev_face_vert_ind;		// contains the indices of faces in the device memory (GPU memory)
int * dev_face_vert_ind_cpy;	// a copy for indices of faces in the device memory (GPU memory)
int * dev_face_vert_ind_cpy2;	// another copy for indices of faces in the device memory (GPU memory)


float * pointers_sums[2];		// stores the index of the pointers_sums which points to the most updated sums array
int ind2_sums;					// the pointer to contain the address of the sums array
float * dev_face_sums;			// it contains the sums of components of each vertex in the device memory (GPU memory)
float * dev_face_sums_cpy;		// a copy for the sums of components of each vertex in the device memory (GPU memory)


/* Local functions */
__global__ void kernel_fill_sums_inds(vertex * vs, float * sums, int * inds, const unsigned int vertices_length);
__device__ void get_first_greatest(float * arr, int len, float a, int * res_fg);
__device__ void get_last_smallest(float * arr, int len, float a, int * res_ls);
__global__ void kernel_update_faces(vertex * f_in, vertex * f_out, int * inds, const unsigned int vertices_length);
/* Local typedefs */
typedef void (*func_ptr_sub_triangle_t)(triangle, vertex *, triangle *);


/*******************************************************************************
 * Function:        cuda_cpy_input_data
 *
 * Description:     This function allocate the memory for various array in the GPU
 *					memory. It also allocates the memory for gpu_out_vertices and 
 *					gpu_out_faces in the CPU memory. It copies the initial icosphere 
 *					data to the dev_faces array in the device.
 *
 * Arguments:       None
 *
 * Return Values:   None
*******************************************************************************/
void cuda_cpy_input_data(){

	CUDA_CALL(cudaMalloc((void **)&dev_faces, faces_length * sizeof(triangle)));
	CUDA_CALL(cudaMalloc((void **)&dev_faces_cpy, faces_length * sizeof(triangle)));

	CUDA_CALL(cudaMalloc((void **)&dev_face_vert_ind, 3*faces_length * sizeof(int)));
	CUDA_CALL(cudaMalloc((void **)&dev_face_vert_ind_cpy, 3*faces_length * sizeof(int)));
	CUDA_CALL(cudaMalloc((void **)&dev_face_vert_ind_cpy2, 3*faces_length * sizeof(int)));

	CUDA_CALL(cudaMalloc((void **)&dev_face_sums, 3*faces_length * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**) &dev_face_sums_cpy, 3*faces_length* sizeof(float)));

	CUDA_CALL(cudaMalloc((void**) &dev_vertices_ico, vertices_length * sizeof(vertex)));

	// copy the initial icosphere data to the device array
	CUDA_CALL(cudaMemcpy(dev_faces, faces_init, ICOSPHERE_INIT_FACE_LEN*sizeof(triangle), cudaMemcpyHostToDevice));


	// set the face pointers
	pointers_faces[0] = dev_faces;		
	pointers_faces[1] = dev_faces_cpy;
	ind2_faces = 0;						// set the index denoting the latest face array to 0

	// set the sum pointers
	pointers_sums[0] = dev_face_sums;
	pointers_sums[1] = dev_face_sums_cpy;
	ind2_sums = 0;						// set the index denoting the latest sum of components of vertices array to 0

	// set the indices pointers
	pointers_inds[0] = dev_face_vert_ind;
	pointers_inds[1] = dev_face_vert_ind_cpy;
	ind2_inds = 0;						// set the index denoting the latest indices array to 0

	gpu_out_faces = (triangle *)malloc(faces_length*sizeof(triangle));
	gpu_out_vertices = (vertex *) malloc(vertices_length*sizeof(vertex));
}

/*******************************************************************************
 * Function:        cuda_cpy_output_data
 *
 * Description:     This function copies the latest faces array to the CPU memory.
 *					It also copies the vertices generated by the GPU to the CPU 
 *					memory.
 *
 * Arguments:       None
 *
 * Return Values:   None
*******************************************************************************/
void cuda_cpy_output_data(){
	CUDA_CALL(cudaMemcpy(gpu_out_faces, pointers_faces[ind2_faces], faces_length*sizeof(triangle), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(gpu_out_vertices, dev_vertices_ico, vertices_length*sizeof(vertex), cudaMemcpyDeviceToHost));
}

/*******************************************************************************
 * Function:        free_gpu_memory
 *
 * Description:     This function frees the GPU memory and the corresponding GPU
 *					output memory in the CPU.
 *
 * Arguments:       None
 *
 * Return Values:   None
*******************************************************************************/
void free_gpu_memory(){
	CUDA_CALL(cudaFree(dev_faces));
	CUDA_CALL(cudaFree(dev_faces_cpy));

	CUDA_CALL(cudaFree(dev_face_vert_ind));
	CUDA_CALL(cudaFree(dev_face_vert_ind_cpy));
	CUDA_CALL(cudaFree(dev_face_vert_ind_cpy2));

	CUDA_CALL(cudaFree(dev_face_sums));
	CUDA_CALL(cudaFree(dev_face_sums_cpy));

	CUDA_CALL(cudaFree(dev_vertices_ico));

	free(gpu_out_faces);
	free(gpu_out_vertices);
}

/*******************************************************************************
 * Function:        break_triangle
 *
 * Description:     This function returns the midpoint of the edges of the input
 *					triangle. The output of the triangle is the list of three 
 *					vertices each being the midpoint of the an edge of the triangle.
 *					triangle tri_i = faces[i];
 *
 *									         P0
 *									        / \
 *									  V[0] *---* V[2]
 *									      / \ / \
 *									    P1---*---P2
 *									         V[1]
 *
 *					Where:
 *					INPUT:	face_tmp.v[0] = P0, face_tmp.v[1] = P1, face_tmp.v[2] = P2
 *					OUTPUT:	v_tmp[0] = V[0], v_tmp[1] = V[1], v_tmp[2] = V[2]
 *
 * Arguments:       triangle face_tmp: The triangle whose edges is to be broken into 
 *										two equal parts
 *					vertex * tmp: The array to which the results will be written.
 *					float radius: The radius of the sphere to which the icosphere
 *									is to be project
 *
 *
 * Return Values:   None
*******************************************************************************/
__device__ void break_triangle(triangle face_tmp, vertex * v_tmp, float radius) {
	float x_tmp, y_tmp, z_tmp, scale;

	// Loop over the three vertices of the triangle
	for(int i=0; i<3; i++){

		// find the midpoint
		x_tmp = (face_tmp.v[i].x + face_tmp.v[(i+1)%3].x)/2;
		y_tmp = (face_tmp.v[i].y + face_tmp.v[(i+1)%3].y)/2;
		z_tmp = (face_tmp.v[i].z + face_tmp.v[(i+1)%3].z)/2;

		// project the point to the sphere
		scale = radius/sqrtf(x_tmp*x_tmp + y_tmp*y_tmp + z_tmp*z_tmp);

		// store the result
		v_tmp[i].x = x_tmp*scale;
		v_tmp[i].y = y_tmp*scale;
		v_tmp[i].z = z_tmp*scale;
	}
}

/*******************************************************************************
 * Function:        sub_triangle_top
 *
 * Description:     This functions stores the trignale P0, V[0], V[2] to the "res". 
 *
 *									         P0
 *									        / \
 *									  V[0] *---* V[2]
 *									      / \ / \
 *									    P1---*---P2
 *									         V[1]
 *
 *					Where:
 *					INPUT:	face_tmp.v[0] = P0, face_tmp.v[1] = P1, face_tmp.v[2] = P2
 *							v_tmp[0] = V[0], v_tmp[1] = V[1], v_tmp[2] = V[2]
 *					OUTPUT:	res.v[0] = P[0], res.v[1] = V[0], res.v[2] = V[2]
 *
 * Arguments:       triangle face_tmp: The triangle which is to be broken.
 *					vertex * v_tmp: The array that contains the middle points of edges
 *									of the triangle.
 *					triangle res: The triangle to which the result is to be written.
 *
 *
 * Return Values:   None
*******************************************************************************/
__device__ void sub_triangle_top(triangle face_tmp, vertex * v_tmp, triangle * res) {
	res->v[0] = face_tmp.v[0];
	res->v[1] = v_tmp[0];
	res->v[2] = v_tmp[2];
}

/*******************************************************************************
 * Function:        sub_triangle_left
 *
 * Description:     This functions stores the trignale V[0], P1, V[1] to the "res". 
 *
 *									         P0
 *									        / \
 *									  V[0] *---* V[2]
 *									      / \ / \
 *									    P1---*---P2
 *									         V[1]
 *
 *					Where:
 *					INPUT:	face_tmp.v[0] = P0, face_tmp.v[1] = P1, face_tmp.v[2] = P2
 *							v_tmp[0] = V[0], v_tmp[1] = V[1], v_tmp[2] = V[2]
 *					OUTPUT:	res.v[0] = V[0], res.v[1] = P1, res.v[2] = V[1]
 *
 * Arguments:       triangle face_tmp: The triangle which is to be broken.
 *					vertex * v_tmp: The array that contains the middle points of edges
 *									of the triangle.
 *					triangle res: The triangle to which the result is to be written.
 *
 *
 * Return Values:   None
*******************************************************************************/
__device__ void sub_triangle_left(triangle face_tmp, vertex * v_tmp, triangle * res) {
	res->v[0] = v_tmp[0];
	res->v[1] = face_tmp.v[1];
	res->v[2] = v_tmp[1];
}

/*******************************************************************************
 * Function:        sub_triangle_right
 *
 * Description:     This functions stores the trignale V[1], P2, V[2] to the "res". 
 *
 *									         P0
 *									        / \
 *									  V[0] *---* V[2]
 *									      / \ / \
 *									    P1---*---P2
 *									         V[1]
 *
 *					Where:
 *					INPUT:	face_tmp.v[0] = P0, face_tmp.v[1] = P1, face_tmp.v[2] = P2
 *							v_tmp[0] = V[0], v_tmp[1] = V[1], v_tmp[2] = V[2]
 *					OUTPUT:	res.v[0] = V[1], res.v[1] = P2, res.v[2] = V[2]
 *
 * Arguments:       triangle face_tmp: The triangle which is to be broken.
 *					vertex * v_tmp: The array that contains the middle points of edges
 *									of the triangle.
 *					triangle res: The triangle to which the result is to be written.
 *
 *
 * Return Values:   None
*******************************************************************************/
__device__ void sub_triangle_right(triangle face_tmp, vertex * v_tmp, triangle * res) {
	res->v[0] = v_tmp[1];
	res->v[1] = face_tmp.v[2];
	res->v[2] = v_tmp[2];
}

/*******************************************************************************
 * Function:        sub_triangle_center
 *
 * Description:     This functions stores the trignale V[0], V[1], V[2] to the "res". 
 *
 *									         P0
 *									        / \
 *									  V[0] *---* V[2]
 *									      / \ / \
 *									    P1---*---P2
 *									         V[1]
 *
 *					Where:
 *					INPUT:	face_tmp.v[0] = P0, face_tmp.v[1] = P1, face_tmp.v[2] = P2
 *							v_tmp[0] = V[0], v_tmp[1] = V[1], v_tmp[2] = V[2]
 *					OUTPUT:	res.v[0] = V[0], res.v[1] = V[1], res.v[2] = V[2]
 *
 * Arguments:       triangle face_tmp: The triangle which is to be broken.
 *					vertex * v_tmp: The array that contains the middle points of edges
 *									of the triangle.
 *					triangle res: The triangle to which the result is to be written.
 *
 *
 * Return Values:   None
*******************************************************************************/
__device__ void sub_triangle_center(triangle face_tmp, vertex * v_tmp, triangle * res) {
	res->v[0] = v_tmp[0];
	res->v[1] = v_tmp[1];
	res->v[2] = v_tmp[2];
}


/* contains the list of functions which breaks a triangle into smaller triangle. */
__device__ func_ptr_sub_triangle_t funcs_list[4] = {sub_triangle_top, sub_triangle_left, sub_triangle_right, sub_triangle_center};


/*******************************************************************************
 * Function:        refine_icosphere_kernel
 *
 * Description:     This is a more optimized kernel which refines the icosphere i.e. 
 * 					increase the depth of the icosphere by one. Let us say we want
 * 					the icosphere to depth "d+1", then the array of faces passed to the
 * 					kernel should contain the icosphere faces corresponding to the depth
 * 					"d". Note that for a depth "d" the size of faces array is:
 * 							size of faces array  = 20*4^(d) * sizeof(triangle).
 * 					Note: Size of "triangle" is 36 bytes. 
 * 					Hence it is important to make sure that the "faces_out" passed to 
 * 					the kernel should already have enough memory allocated to it.
 *
 *					The array "faces" contains the faces of icosphere corresponding to the 
 *					depth "d".
 *
 *					In this kernel four threads operate on one face and generate 4 new 
 *					faces and stores the result in the "faces_out" array passed to it.
 *
 *					Say a thread "i" is working on the faces[i]. Then the thread "i"
 *					finds the mid points of the triangle "faces[i]" as shown in the following
 *					diagram.
 *
 *									         P0
 *									        / \
 *									  V[0] *---* V[2]
 *									      / \ / \
 *									    P1---*---P2
 *									         V[1]
 *
 *					Where:
 *							P0 = faces[i].v[0], P1 = faces[i].v[1], P2 = faces[i].v[2]
 * 					Now thread "i" stores the result back to the "faces_out" depending on its
 *					thread index.
 * 					If i%4 == 0 then the thread stores the P0, V[0], V[2] triangle to faces_out[i].
 * 					else if i%4 == 1 then the thread stores the V[0], P1, V[1] triangle to faces_out[i].
 * 					else if i%4 == 2 then the thread stores the V[2], P2, V[0] triangle to faces_out[i].
 * 					else if i%4 == 3 then the thread stores the V[0], V[1], V[2] triangle to faces_out[i].
 *
 * 					It is to be noted that the structure of the kernel is made in a way that we don't
 * 					have any "IF" statement in the kernel hence avoiding the branching in the warp. Also,
 *					we are not using the threads in the "Y" direction i.e. threadIdx.y to keep the memory
 *					access coalesced.
 *
 * Arguments:       triangle faces: The array of faces for depth 'd'
 *					triangle faces_out: The array of faces to which the output is written
 *					const float radius: The radius of the sphere
 *					const unsigned int th_len: The length of the "faces" array.
 *
 * Return Values:   None
*******************************************************************************/
__global__ void refine_icosphere_kernel(triangle * faces, triangle * faces_out, const float radius, const unsigned int th_len) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;

	vertex v_tmp[3];
	triangle v;

	while(idx < 4*th_len){
		int tri_ind = idx/4;
		int sub_tri_ind = idx%4;
		v = faces[tri_ind];

		// break the triangle
		break_triangle(v, v_tmp, radius);

		// store the result appropriately
		funcs_list[sub_tri_ind](v, v_tmp, &faces_out[idx]);

		idx += numthrds;
	}

}

/*******************************************************************************
 * Function:        cudacall_icosphere
 *
 * Description:     This calls the optimized kernel repeatedly to generate the icosphere
 * 					of depth "max_depth". Once the icosphere generation is over then 
 * 					a kernel is called to fills the "sums" array (contains the sum of
 * 					components of each vertex) and the "indices" arraY (contains the
 * 					indices of each vertex in the faces array). The functions
 * 					finds out the number of blocks required with a upper limit of 
 * 					65535.
 *
 * Arguments:       int  thread_num: No. of thread per block
 *
 * Return Values:   None
*******************************************************************************/
void cudacall_icosphere(int thread_num) {
	// each thread creates a sub triangle
	int ths, n_blocks, ind1;
	for(int i=0; i<max_depth; i++){
		ths = 20*pow(4, i);
		n_blocks = std::min(65535, (ths + 4*thread_num  - 1) / thread_num);
		ind1 = i%2;
		ind2_faces = (i+1)%2;
		refine_icosphere_kernel<<<n_blocks, thread_num>>>(pointers_faces[ind1], pointers_faces[ind2_faces], radius, ths);
	}
	int len = 3*faces_length;
	n_blocks = std::min(65535, (len + thread_num  - 1) / thread_num);
	kernel_fill_sums_inds<<<n_blocks, thread_num>>>((vertex *)pointers_faces[ind2_faces], dev_face_sums, dev_face_vert_ind, len);
}

/*******************************************************************************
 * Function:        kernel_fill_sums_inds
 *
 * Description:     This is a simple kernel to fill the "sums" array with the 
 * 					sum of components of each vertex and "inds" array with the 
 *					indices of the vertex array.
 *
 *					A thread "i" operates on one vertex and perform following
 *					operatoin
 *						sums[i] = vs[i].x + vs[i].y + vs[i].z
 *						inds[i] = i
 *
 *					The "sums" and "inds" array are required for sorting and
 *					removing the duplicate arrays.
 *
 * Arguments:       vertex * vs: The array of vertices obtained from list of faces.
 * 					int * sum: The array to contain the sum of components of vertices "vs".
 * 					int * inds: The array to contain the indices corresponding to "vs".
 * 					const unsigned int vertices_length: THe length of the vertices length.
 *
 * Return Values:   None
*******************************************************************************/
__global__
void kernel_fill_sums_inds(vertex * vs, float * sums, int * inds, const unsigned int vertices_length){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;

	while(idx < vertices_length){
		sums[idx] = vs[idx].x + vs[idx].y + vs[idx].z;
		inds[idx] = idx;
		idx += numthrds;
	}
}

/*******************************************************************************
 * Function:        dev_merge
 *
 * Description:     This function SEQUENTIALLY merges two already sorted arrays into
 * 					a single sorted array. The two arrays here are:
 *
 * 					s[idx] and s[start]. Let us say arr1 denotes s[idx] array and
 * 					arr2 denotes s[start] array.
 *
 * 					The size of arr1 = start - idx
 * 					The size of arr2 = end - start
 *
 * 					Following is the reference to the merging of sorted arrays into
 * 					one sorted array:
 * 					https://www.geeksforgeeks.org/merge-two-sorted-arrays/
 *
 * 					The arr1 and arr2 is sorted and the result is put into the 
 * 					array "r".
 *
 * 					Note that the array "ind[start]" and "ind[idx]" is also sorted but
 * 					with respect to the "s[idx]" and "s[start]" array. That is the values
 * 					of "s[idx]" and "s[start]" are used for comparison and "ind[idx]" and
 * 					"ind[start]" are sorted accordingly.
 *
 * Arguments:       float * s: The array of sums of vertices
 * 					float * r: The array to which the sorted sums array will be stored
 * 					int * ind: The array of indices of vertices
 * 					int * ind_res: The array to which the sorted indices array will be saved.
 *					unsigned int idx: The start of first array
 *					unsigned int start: The start of second array
 *					unsinged int end: The end of second array
 *
 * Return Values:   None
*******************************************************************************/
__device__
void dev_merge(float * s, float * r, int * ind, int * ind_res, unsigned int idx, unsigned int start, unsigned int end){
	unsigned int c=idx;
	unsigned int i=idx;unsigned int j=start;
	while(j<end && i<start){
		if(s[i] <= s[j]){
			r[c] = s[i];
			ind_res[c] = ind[i];
			i++;
		}
		else{
			r[c] = s[j];
			ind_res[c] = ind[j];
			j++;
		}
		c++;
	}
	while(i < start){
		r[c] = s[i];
		ind_res[c] = ind[i];
		c++;i++;
	}

	while(j < end){
		r[c] = s[j];
		ind_res[c] = ind[j];
		c++;j++;
	}
}

/*******************************************************************************
 * Function:        kernel_merge_sort
 *
 * Description:     This is a naive kernal which implements the a step of merge sorting. The
 * 					input "r" represents the total length of the two arrays. For 
 * 					example consider the following example:
 *
 * 					sums[] = {4, 3, 2, 1, 0};
 *
 * 					We have ceil(log2(length(sums))) = 3, hence the kernel has to 
 * 					be called 3 times for ith time (starting i=0) the value of
 * 					"r" should be 2^(i+1).
 *
 * 					So when kernel is called:
 * 					For r = 2 (iteration 0);
 * 						Thread 0 works on arrays partitions [4] [3]
 * 						Thread 1 works on arrays partitions [2] [1]
 *
 * 						Result stored in "res" array:
 * 						[3, 4, 1, 2, 0];
 *
 * 					For r = 4 (iteration 1)
 * 						Thread 0 works on arrays partitions [3, 4] [1, 2]
 *
 *						Result stored in "res" array:
 * 						[1, 2, 3, 4, 0];
 *
 * 					for r = 8 (iteration 2)
 * 						Thread 0 works on arrays partitions [1, 2, 3, 4] [0]
 *
 *						Result stored in "res" array:
 * 						[0, 1, 2, 3, 4];	
 *
 * 					This kernel perform one step of the merge sort on chuncks of 1024 elements
 * 					by copying the elements to the shared memory and then copying the results
 * 					back to the global memory
 *
 * Arguments:       float * sums: The array of sums of vertices
 * 					float * res: The array to which the sorted sums array will be stored
 * 					int * ind: The array of indices of vertices
 * 					int * ind_res: The array to which the sorted indices array will be saved.
 *					const unsigned int length: The total length of the array
 *					const unsigned int r: Equals to 2 times of the length of sub-arrays which have to be sorted.
 *
 * Return Values:   None
*******************************************************************************/
__global__
void kernel_merge_sort(float * sums, float * res, int * ind, int * ind_res, const unsigned int length, const unsigned int r){

	__shared__ float sh_sums[1024];
	__shared__ float sh_res[1024];
	__shared__ int sh_ind[1024];
	__shared__ int sh_indres[1024];
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;

	const int stride = r/2;

	int id = threadIdx.x;
	int t_len = min(1024, length - blockIdx.x * blockDim.x);

	while(idx < length){
		// copy to shared mem
		sh_sums[threadIdx.x] = sums[idx];
		sh_ind[threadIdx.x] = ind[idx];

		__syncthreads();

		// perform a step of merge sort
		if(id%r == 0)
			dev_merge(sh_sums, sh_res, sh_ind, sh_indres, id, min(t_len, id + stride), min(t_len, id+r));

		__syncthreads();

		// copy result to global mem
		res[idx] = sh_res[threadIdx.x];
		ind_res[idx] = sh_indres[threadIdx.x];
		
		__syncthreads();
		
		idx += numthrds;
	}
}

/*******************************************************************************
 * Function:        kernel_merge_chuncks
 *
 * Description:     This is a kernal which implements PARALLELED merging of sorted
 * 					arrays. 
 * 					The Algorithm 1 of following reference describes the 
 * 					parallel merging of sorted arrays:
 *
 * 					: http://www2.hawaii.edu/~nodari/teaching/f16/notes/notes10.pdf
 *
 * 					The above reference assumes that the arrays don't contain the
 * 					duplicate elements. In our situation, we are sorting the arrays
 * 					to remove the duplicate elements hence we cannot use the above
 * 					algorithm directly.
 *
 * 					We have come up with a new algorithm which is a modified version
 * 					of the above algorithm. Following is the description of the 
 * 					algorithm.
 *
 * 					Say we have two sorted arrays, arr1 and arr2 to be merged and
 * 					have duplicate elements. The result of merging arr1 and arr2 has 
 * 					to be put in arr_res array.
 *
 * 		Task	1.	For an element "i" in arr1 i.e. arr1[i], find the index of largest
 * 					number in arr2 which is smaller than arr1[i] i.e.
 *
 * 						LS_index = argmax_j (arr2[j] < arr1[i]).
 *
 * 					LS_index is the index of the largest number in arr2 such that 
 * 					arr2[LS_index] < arr1[i]. Now place the arr1[i] at position
 * 					i + LS_index + 1 in the array arr_res i.e.
 *
 * 						arr_res[i+LS_index+1] = arr1[i].
 *
 * 					Do the task 1 for every element of arr1.
 *
 * 		Task	2.	Now for an elemetn "i" i arr2 i.e. arr2[i], find the index of smallest
 * 					number in arr1 which is larger than arr2[o] i.e.
 *
 * 						SL_index = argmin_j (arr2[i] < arr1[j])
 *
 * 					SL_index is the index of smallest number in arr1 such that
 * 					arr2[i] < arr1[SL_index]. Now place the arr2[i] at position i + SL_index
 * 					in the array arr_res i.e.
 *
 * 						arr_res[i+SL_index] = arr2[2].
 *
 * 					Do the task 2 for every element of arr2.
 *
 * 					The above algorithm has been parallelized with each threads operating
 * 					at each element of the array. Let us say we have two arrays, each of size
 *					1024 elements, then total of 2048 threads are used to merge the two arrays.
 *
 * Arguments:       float * sums: The array of sums of vertices
 * 					float * res: The array to which the sorted sums array will be stored
 * 					int * ind: The array of indices of vertices
 * 					int * ind_res: The array to which the sorted indices array will be saved.
 *					const unsigned int length: The total length of the array
 *					const unsigned int r: Equals to 2 times of length of sub-arrays which have to be sorted.
 *
 * Return Values:   None
*******************************************************************************/
__global__
void kernel_merge_chuncks(float * sums, float * res, int * ind, int * ind_res, const unsigned int length, const unsigned int r){
	
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;
	const int stride = r/2;
	
	int tmp_res[1];
	
	int k;
	int local_k;
	int arr_len;
	int arr_start;
	int arr_ind_L, arr_ind_HE, final_index;
	
	while(idx < length){
		k = idx % r;
		local_k = k%stride;
		arr_ind_L = idx - local_k + stride;    // arr2 
		arr_ind_HE = idx - local_k - stride;   // arr1 

		if(k < stride && arr_ind_L < length){
			// an arr 1 element
			arr_len = min(stride, length - arr_ind_L);
			arr_start = idx - local_k + 1;

			get_last_smallest(&sums[arr_ind_L], arr_len, sums[idx], tmp_res);

			final_index = local_k + tmp_res[0] + arr_start;

		}else if( k>=stride && 0 <= arr_ind_HE){
			// an arr 2 element
			arr_len = min(stride, length - arr_ind_HE);
			arr_start = idx - local_k - stride;

			get_first_greatest(&sums[arr_ind_HE], arr_len, sums[idx], tmp_res);
			
			final_index = local_k + tmp_res[0] + arr_start;
		}
		
		// now place the element
		res[final_index] = sums[idx];
		ind_res[final_index] = ind[idx];
		
		idx += numthrds;
	}

}

/*******************************************************************************
 * Function:        cudacall_fill_vertices
 *
 * Description:     This calls the optimized sorting algorithms till ceil(log2(length(vertices))).
 * 					Once the sorting is complete it calls the "cuda_remove_duplicates" function
 * 					to remove the duplicate vertices from the sorted.
 *
 * 					Note that we are not sorting the arrays of vertices obtained from the 
 * 					faces array with respect to the sum of their components, rather we are
 * 					sorting the "indices" arrays with respect to the sum of components of 
 * 					the vertices. Once the sorting is complete the array of vertices is
 * 					updated with respect to the sorted indices array.
 *
 * 					The above approach allow the faster access to global memory and allows us
 * 					to use shared memory as the size of the sorting arrays has decreased
 * 					considerably.
 *
 * 					The order the sequential implementation of this algorithm is: O(Log(n)^2)
 * 					The order the parallel implementation of this algorithm is: O(Log(n)^2/m)
 * 					where m: no. of parallel running processors
 *
 * 					Hence this method performs much better than "naive" implementation.
 *
 * Arguments:       int thread_num: The number of threads per block.
 *
 * Return Values:   None
*******************************************************************************/
void cudacall_fill_vertices(int thread_num) {
	
	unsigned int len = 3*faces_length;
	int n_blocks = min(65535, (len + thread_num  - 1) / thread_num);

	unsigned int l = ceil(log2(thread_num)), ind1;
	for(int i=0; i<l; i++){
		ind1 = i%2;
		ind2_sums = (i+1)%2;
		ind2_inds = ind2_sums;
		unsigned int r = pow(2, i+1);
		kernel_merge_sort<<<n_blocks, thread_num>>>(pointers_sums[ind1], pointers_sums[ind2_sums], pointers_inds[ind1], pointers_inds[ind2_inds], len, r);

	}

	// now sort the chunks of 1024 floats
	l = ceil(log2(n_blocks));
	for(int i=0; i<l; i++){
		ind1 = (ind1+1)%2;
		ind2_sums = (ind2_sums+1)%2;
		ind2_inds = ind2_sums;
		unsigned int r = pow(2, i+1)*1024;
		kernel_merge_chuncks<<<n_blocks, thread_num>>>(pointers_sums[ind1], pointers_sums[ind2_sums], pointers_inds[ind1], pointers_inds[ind2_inds], len, r);
	}

	int out = (ind2_faces + 1) %2;
	kernel_update_faces<<<n_blocks, thread_num>>>((vertex *)pointers_faces[ind2_faces], (vertex *)pointers_faces[out], pointers_inds[ind2_inds], len);
	ind2_faces = out;
}


/*******************************************************************************
 * Function:        get_first_greatest
 *
 * Description:     This kernel finds the index of the smallest value in the array
 * 					which is greater than the value "a" passed to it i.e. it finds
 * 					the index first greatest number.
 *
 * 					Mathematically:
 *
 *  					ref_fg = argmin_j (a < arr2[j]).
 *
 * Arguments:       floar * arr: The array in which the first greatest is to be found
 * 					int len: The length of the array
 * 					float a: The value with respect to which the first greatest has 
 * 								to be found
 * 					int * res_fg: It is used to return the result
 *
 * Return Values:   None
*******************************************************************************/
__device__
void get_first_greatest(float * arr, int len, float a, int * res_fg){
	int first = 0, last = len - 1;
	while (first <= last)
	{
		int mid = (first + last) / 2;
		if (arr[mid] > a)
			last = mid - 1;
		else
			first = mid + 1;
	}
	res_fg[0] =  last + 1 == len ? len : last + 1;

}

/*******************************************************************************
 * Function:        get_last_smallest
 *
 * Description:     This kernel finds the index of the largest value in the array
 * 					which is smaller than the value "a" passed to it i.e. it finds
 * 					the index first greatest number.
 *
 * 					Mathematically:
 *
 *  					ref_ls = argmax_j (arr2[j] < a).
 *
 * Arguments:       floar * arr: The array in which the last smallest is to be found
 * 					int len: The length of the array
 * 					float a: The value with respect to which the last smallest has 
 * 								to be found
 * 					int * res_fg: It is used to return the result
 *
 * Return Values:   None
*******************************************************************************/
__device__
void get_last_smallest(float * arr, int len, float a, int * res_ls){
	int first = 0, last = len - 1;
	while (first <= last)
	{
		int mid = (first + last) / 2;
		if (arr[mid] >= a)
			last = mid - 1;
		else
			first = mid + 1;
	}
	res_ls[0] = first - 1 < 0 ? -1 : first - 1;
}

/*******************************************************************************
 * Function:        kernel_update_faces
 *
 * Description:     This kernel updates the vertices array based on the indices
 					array passed to it. 
 *
 * Arguments:       vertex * f_in: The input vertices array
 * 					vertex * f_out: The output vertices arraY
 * 					int * inds: The indices array as per which the output array
 * 									will be written to.
 * 					int vertice_length: The lenght of f_in, f_out and inds arrays
 *
 * Return Values:   None
*******************************************************************************/
__global__
void kernel_update_faces(vertex * f_in, vertex * f_out, int * inds, const unsigned int vertices_length){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;

	while(idx < vertices_length){
		f_out[idx] = f_in[inds[idx]];
		inds[idx] = idx;
		idx += numthrds;
	}
}