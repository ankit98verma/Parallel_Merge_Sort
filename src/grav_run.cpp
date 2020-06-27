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
#include "ta_utilities.hpp"

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
 * Function:        time_profile_cpu
 *
 * Description:     RUNS the CPU code
 *
 * Arguments:       bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   CPU computational time
*******************************************************************************/
void time_profile_cpu(bool verbose, float * res){

	float cpu_time_icosphere_ms = 0;
	float cpu_time_fill_vertices_ms = 0;

	START_TIMER();
		create_icoshpere();
	STOP_RECORD_TIMER(cpu_time_icosphere_ms);

	START_TIMER();
		fill_vertices();
	STOP_RECORD_TIMER(cpu_time_fill_vertices_ms);


    if(verbose){
	    printf("Icosphere generation time: %f ms\n", cpu_time_icosphere_ms);
	    printf("Fill vertices time: %f ms\n", cpu_time_fill_vertices_ms);
    }
    res[0] = cpu_time_icosphere_ms+cpu_time_fill_vertices_ms;
    res[1] = 0;
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
void time_profile_gpu(bool verbose, float * res){

	float gpu_time_icosphere = 0, gpu_time_fill_vertices = 0;
	float gpu_time_indata_cpy = 0;
	float gpu_time_outdata_cpy = 0;

	cudaError err;

	START_TIMER();
		cuda_cpy_input_data();
	STOP_RECORD_TIMER(gpu_time_indata_cpy);


	START_TIMER();
		cudacall_icosphere(ICOSPHERE_GPU_THREAD_NUM);
	STOP_RECORD_TIMER(gpu_time_icosphere);
	err = cudaGetLastError();
    if (cudaSuccess != err){
        cerr << "Error " << cudaGetErrorString(err) << endl;
    }else{
    	if(verbose)
        	cerr << "No kernel error detected" << endl;
    }

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

	res[0] = gpu_time_icosphere + gpu_time_fill_vertices + gpu_time_outdata_cpy + gpu_time_indata_cpy;
	res[1] = 0;
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
void run(int depth, float radius, bool verbose, float * cpu_res, float * gpu_res){

	init_vars(depth, radius);
	allocate_cpu_mem(verbose);
	init_icosphere();

	cpu_res[0] = 0;
	cpu_res[1] = 0;

#if defined(CPU_ONLY) || defined(CPU_GPU_ONLY)
	if(verbose)
		cout << "\n----------Running CPU Code----------\n" << endl;
	time_profile_cpu(verbose, cpu_res);
#endif

#if defined(GPU_ONLY) || defined(CPU_GPU_ONLY)
	if(verbose)
		cout << "\n----------Running GPU Code----------\n" << endl;
	time_profile_gpu(verbose, gpu_res);
#endif

#ifdef CPU_GPU_ONLY
	float cpu_time = cpu_res[0] +  cpu_res[1];
	float gpu_time = gpu_res[0] +  gpu_res[1];
	if(verbose){
		cout << "\nTime taken by the CPU is: " << cpu_time << " milliseconds" << endl;
		cout << "Time taken by the GPU is: " << gpu_time << " milliseconds" << endl;
		cout << "Speed up factor: " << cpu_time/gpu_time << "\n" << endl;
	}

	// calculate the distance b/w two points of icosphere
	float norm0 = faces[0].v[0].x*faces[0].v[0].x + faces[0].v[0].y*faces[0].v[0].y + faces[0].v[0].z*faces[0].v[0].z;
	float norm1 = faces[0].v[1].x*faces[0].v[1].x + faces[0].v[1].y*faces[0].v[1].y + faces[0].v[1].z*faces[0].v[1].z;
	float ang = acosf((faces[0].v[0].x*faces[0].v[1].x + faces[0].v[0].y*faces[0].v[1].y + faces[0].v[0].z*faces[0].v[1].z)/(norm1*norm0));
	float dis = radius*ang;
	if(verbose)
		cout << "Distance b/w any two points of icosphere is: " << dis << " (unit is same as radius)\n" << endl;
#endif
	
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

	float cpu_times[2],gpu_times[2];

	float r = 1;
	run(len, r, verbose, cpu_times, gpu_times);

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

	cout << "Exporting: gpu_sorted_vertices.csv"<<endl;

	string filename1 = "results/gpu_sorted_vertices.csv";
	ofstream obj_stream;
	obj_stream.open(filename1);
	obj_stream << "x, y, z" << endl;
	vertex * v = (vertex *) gpu_out_faces;
	cout <<"-----------------------" << endl;
	for(unsigned int i=0; i< 3*faces_length; i++){
		obj_stream << v[i].x << ", " << v[i].y << ", " << v[i].z << endl;
	}
	obj_stream.close();

    // cout << "Exporting: gpu_vertices.csv"<<endl;

    // string filename2 = "results/gpu_vertices.csv";
    // ofstream obj_stream2;
    // obj_stream2.open(filename2);
    // obj_stream2 << "x, y, z" << endl;
    // cout <<"-----------------------" << endl;
    // for(unsigned int i=0; i< vertices_length; i++){
    //     obj_stream2 << gpu_out_vertices[i].x <<", "<< gpu_out_vertices[i].y <<", "<< gpu_out_vertices[i].z  << endl;
    // }
    // obj_stream2.close();
}

/*******************************************************************************
 * Function:        export_cpu_outputs
 *
 * Description:     Exports the cpu_vertices, cpu_edges, and cpu_potentials
 *
 * Arguments:       bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   none
*******************************************************************************/
void export_cpu_outputs(bool verbose){

	cout << "Exporting: cpu_vertices.csv"<<endl;

	string filename1 = "results/cpu_vertices.csv";
	ofstream obj_stream;
	obj_stream.open(filename1);
	obj_stream << "x, y, z" << endl;
	cout <<"-----------------------" << endl;
	for(unsigned int i=0; i< vertices_length; i++){
		obj_stream << vertices[i].x << ", " << vertices[i].y << ", " << vertices[i].z << endl;
	}
	obj_stream.close();

    cout << "Exporting: cpu_edges.csv"<<endl;

    ofstream obj_stream2;
	obj_stream2.open("results/cpu_edges.csv");
	obj_stream2 << "x1, y1, z1, x2, y2, z2" << endl;
	cout <<"-----------------------" << endl;
	for(unsigned int i=0; i<3*faces_length; i++){
		triangle triangle_tmp = faces[i];
		for(int j=0; j<3;j++)
			obj_stream2 << 	triangle_tmp.v[j].x << ", " << triangle_tmp.v[j].y << ", " << triangle_tmp.v[j].z << ", " <<
							triangle_tmp.v[(j+1)%3].x << ", " << triangle_tmp.v[(j+1)%3].y << ", " << triangle_tmp.v[(j+1)%3].z << endl;
	}
	obj_stream2.close();

    
    if(verbose)
   		cout<<"Exporting: results/cpu_output_potential.mat" << endl;

    std::ofstream f;
    f.open("results/cpu_output_potential.mat", std::ios::out);

    for (unsigned int i=0; i<vertices_length; i++){
        f<<i<<'\t'<<vertices[i].x<<'\t'<<vertices[i].y<<'\t'<<vertices[i].z<<'\t'<<potential[i]<<'\n';
    }
    f.close();

}