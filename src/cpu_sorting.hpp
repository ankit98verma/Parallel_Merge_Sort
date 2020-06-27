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

// typedef long long int intL;
// struct vertex
// {
//     float x;
//     float y;
//     float z;
// };
// typedef struct vertex vertex;

// struct triangle
// {
//     vertex v[3];
// };
// typedef struct triangle triangle;

/**** Variables related to Icosphere ****/

int partition_sum(void * arr, int low, int high);

GLOBAL unsigned int faces_length;

// vertices of the icosphere
GLOBAL unsigned int vertices_length;

// The depth of the icosphere
GLOBAL unsigned int max_depth;

void init_vars(unsigned int depth, float r);

void quickSort(void * arr, int low, int high, int partition_fun(void *, int, int));

void free_cpu_memory();


#endif // CUDA_HEADER_CUH_