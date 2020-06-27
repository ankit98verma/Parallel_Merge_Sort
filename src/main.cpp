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

#include "cuda_sorting.cuh"

using namespace std;


#define ICOSPHERE_GPU_THREAD_NUM		1024

void export_gpu_outputs(bool verbose);
void export_cpu_outputs(bool verbose);

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

	float gpu_time_fill_vertices = 0;
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

    sort_cpu_arr();

    START_TIMER();
		cuda_cpy_output_data();
	STOP_RECORD_TIMER(gpu_time_outdata_cpy);
	if(verbose){
		printf("GPU Input data copy time: %f ms\n", gpu_time_indata_cpy);
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
 *                  bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   none
*******************************************************************************/
void run(int len, bool verbose){

	init_vars(len);

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

	bool verbose = (bool)atoi(argv[2]);
	
	if(verbose)
		cout << "Verbose ON" << endl;
	else
		cout << "Verbose OFF" << endl;

	run(len, verbose);

	export_gpu_outputs(verbose);

	export_cpu_outputs(verbose);

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

    cout << "Exporting: gpu_arr.csv"<<endl;

    string filename2 = "results/gpu_arr.csv";
    ofstream obj_stream2;
    obj_stream2.open(filename2);
    obj_stream2 << "sums" << endl;
    cout <<"-----------------------" << endl;
    for(unsigned int i=0; i< faces_length; i++){
        obj_stream2 << gpu_out_sums[i] << endl;
    }
    obj_stream2.close();
}


void export_cpu_outputs(bool verbose){

    cout << "Exporting: cpu_arr.csv"<<endl;

    string filename2 = "results/cpu_arr.csv";
    ofstream obj_stream2;
    obj_stream2.open(filename2);
    obj_stream2 << "sums" << endl;
    cout <<"-----------------------" << endl;
    for(unsigned int i=0; i< faces_length; i++){
        obj_stream2 << cpu_arr[i] << endl;
    }
    obj_stream2.close();
}
