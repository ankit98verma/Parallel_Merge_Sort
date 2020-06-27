/*
 * grav_run: Ankit Verma, Garima Aggarwal, 2020
 *
 * This file runs the CPU implementation and GPU implementation
 * of the gravitational field calculation.
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>

#include "grav_cuda.cuh"

#include "grav_run.hpp"

using namespace std;

// (From Eric's code)
cudaEvent_t start;
cudaEvent_t stop;
#define START_TIMER() {                         \
      CUDA_CALL(cudaEventCreate(&start));       \
      CUDA_CALL(cudaEventCreate(&stop));        \
      CUDA_CALL(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      CUDA_CALL(cudaEventRecord(stop));                     \
      CUDA_CALL(cudaEventSynchronize(stop));                \
      CUDA_CALL(cudaEventElapsedTime(&name, start, stop));  \
      CUDA_CALL(cudaEventDestroy(start));                   \
      CUDA_CALL(cudaEventDestroy(stop));                    \
    }


/*******************************************************************************
 * Function:        chech_args
 *
 * Description:     Checks for the user inputs arguments to run the file
 *
 * Arguments:       int argc, char argv
 *
 * Return Values:   0
*******************************************************************************/
int check_args(int argc, char **argv){
	if (argc != 3){
        // printf("Usage: ./grav [depth] [thread_per_block] \n");
        printf("Usage: ./grav [depth] [verbose: 0/1]\n");
        return 1;
    }
    return 0;
}

/*******************************************************************************
 * Function:        time_profile_gpu
 *
 * Description:     RUNS the GPU code
 *
 * Arguments:       bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   GPU computational time
*******************************************************************************/
void time_profile_gpu(bool verbose){

	float gpu_time_icosphere = 0, gpu_time_fill_vertices = 0;
	float gpu_time_indata_cpy = 0;
	float gpu_time_outdata_cpy = 0;

	cudaError err;

	START_TIMER();
		cuda_cpy_input_data();
	STOP_RECORD_TIMER(gpu_time_indata_cpy);


	START_TIMER();
		cudacall_fill_vertices(ICOSPHERE_GPU_THREAD_NUM);
	STOP_RECORD_TIMER(gpu_time_fill_vertices);
    err = cudaGetLastError();
    if (cudaSuccess != err){
        cerr << "Error " << cudaGetErrorString(err) << endl;
    }else{
    	if(verbose)
        	cerr << "No kernel error detected" << endl;
    }

    START_TIMER();
		cuda_cpy_output_data();
	STOP_RECORD_TIMER(gpu_time_outdata_cpy);
	if(verbose){
		printf("GPU Input data copy time: %f ms\n", gpu_time_indata_cpy);
	    printf("GPU Icosphere generation time: %f ms\n", gpu_time_icosphere);
	    printf("GPU Fill vertices: %f ms\n", gpu_time_fill_vertices);
		printf("GPU Output data copy time: %f ms\n", gpu_time_outdata_cpy);
	}
}

/*******************************************************************************
 * Function:        run
 *
 * Description:     Stores the vertices and corresponding potential in a MATLAB
 *                  compatible .mat file
 *
 * Arguments:       int depth - needed for icosphere calculation
 *                  float radius - radius of the sphere
 *                  bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   none
*******************************************************************************/
void run(int depth, float radius, bool verbose){

	init_vars(depth, radius);

	if(verbose)
		cout << "\n----------Running GPU Code----------\n" << endl;
	time_profile_gpu(verbose);

	
	free_cpu_memory();

}


/*******************************************************************************
 * Function:        main
 *
 * Description:     Run the main function
 *
 * Arguments:       int argc, char argv
 *
 * Return Values:   int 1 if code executes successfully else 0.
*******************************************************************************/
int main(int argc, char **argv) {

	if(check_args(argc, argv))
		return 0;


	int len = atoi(argv[1]);
	int thres = 10;

#ifdef GPU_ONLY
	thres = 12;
#endif

	if(len >= thres){
		cout << "Depth should be less than " << thres << endl;
		cout << "Exiting! " << endl;
		return -1;
	}

	bool verbose = (bool)atoi(argv[2]);
	
	if(verbose)
		cout << "Verbose ON" << endl;
	else
		cout << "Verbose OFF" << endl;

	float r = 1;
	run(len, r, verbose);

	export_gpu_outputs(verbose);

    return 1;
}

/*******************************************************************************
 * Function:        export_gpu_outputs
 *
 * Description:     Exports the gpu_vertices, gpu_sorted_vertices, and gpu_potentials
 *
 * Arguments:       bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   none
*******************************************************************************/
void export_gpu_outputs(bool verbose){

	// cout << "Exporting: gpu_sorted_vertices.csv"<<endl;

	// string filename1 = "results/gpu_sorted_vertices.csv";
	// ofstream obj_stream;
	// obj_stream.open(filename1);
	// obj_stream << "x, y, z" << endl;
	// vertex * v = (vertex *) gpu_out_faces;
	// cout <<"-----------------------" << endl;
	// for(unsigned int i=0; i< 3*faces_length; i++){
	// 	obj_stream << v[i].x << ", " << v[i].y << ", " << v[i].z << endl;
	// }
	// obj_stream.close();



    cout << "Exporting: gpu_sums.csv"<<endl;

    string filename2 = "results/gpu_sums.csv";
    ofstream obj_stream2;
    obj_stream2.open(filename2);
    obj_stream2 << "sums" << endl;
    cout <<"-----------------------" << endl;
    for(unsigned int i=0; i< 3*faces_length; i++){
        obj_stream2 << gpu_out_sums[i] << endl;
    }
    obj_stream2.close();
}
