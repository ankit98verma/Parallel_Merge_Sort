/*
 * grav_cuda.cuh
 * Kevin Yuh, 2014
 * Revised by Nailen Matschke, 2016
 * Revised by Loko Kung, 2018
 */

#ifndef _GRAV_CUDA_CUH_
#define _GRAV_CUDA_CUH_


#include "device_launch_parameters.h"
#include "cpu_sorting.hpp"
#include "cuda_calls_helper.h"

#undef  GLOBAL
#ifdef _GRAV_CUDA_ICOSPHERE_CU_
#define GLOBAL
#else
#define GLOBAL  extern
#endif


GLOBAL float * gpu_out_sums;

void cuda_cpy_input_data();

void cudacall_fill_vertices(int);

void cuda_cpy_output_data();

void free_gpu_memory();

#endif