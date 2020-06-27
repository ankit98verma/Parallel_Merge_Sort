/*
 * CPU implementation of gravitational field calculator
 * header file.
 */


#ifndef _GRAV_CPU_H_
#define _GRAV_CPU_H_


#include <cstdio>
#include <cstdlib>
#include <iostream>

using std::string;

#undef  GLOBAL
#ifdef _GRAV_CPU_C_
#define GLOBAL
#else
#define GLOBAL  extern
#endif



/**** Variables related to Icosphere ****/

GLOBAL unsigned int faces_length;
GLOBAL float * cpu_arr;

void init_vars(unsigned int depth);

void sort_cpu_arr();

void quickSort(float * arr, int low, int high, int partition_fun(float *, int, int));

void free_cpu_memory();


#endif // CUDA_HEADER_CUH_