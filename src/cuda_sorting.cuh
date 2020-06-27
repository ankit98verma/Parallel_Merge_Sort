/*
 * grav_cuda.cuh
 * Kevin Yuh, 2014
 * Revised by Nailen Matschke, 2016
 * Revised by Loko Kung, 2018
 */

#ifndef _GRAV_CUDA_CUH_
#define _GRAV_CUDA_CUH_


#include "device_launch_parameters.h"
#include "cuda_calls_helper.h"

#undef  GLOBAL
#ifdef _GRAV_CUDA_ICOSPHERE_CU_
#define GLOBAL
#else
#define GLOBAL  extern
#endif


GLOBAL int * gpu_out_arr;

GLOBAL unsigned int arr_len;
GLOBAL int * cpu_arr;

void cuda_cpy_input_data(int * in_arr, unsigned int length);

void cudacall_merge_sort(int);

void cuda_cpy_output_data(int * out_arr, unsigned int length);

void free_gpu_memory();

#endif
