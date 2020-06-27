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

#define	PI			3.1415926f
#define EPSILON     1E-6
#define EPSILON_POT 1E-7
#define ICOSPHERE_INIT_FACE_LEN		20

#define R_eq    		1
#define mhu 			1 // in km^3/s^2
#define N_SPHERICAL 	20
#define N_coeff 		(N_SPHERICAL+1)*(N_SPHERICAL+2)/2

typedef long long int intL;
struct vertex
{
    float x;
    float y;
    float z;
};
typedef struct vertex vertex;

struct triangle
{
    vertex v[3];
};
typedef struct triangle triangle;

/**** Variables related to Icosphere ****/

int partition_sum(void * arr, int low, int high);

GLOBAL unsigned int faces_length;

// vertices of the icosphere
GLOBAL unsigned int vertices_length;

// The depth of the icosphere
GLOBAL unsigned int max_depth;



void init_vars(unsigned int depth, float r);
void allocate_cpu_mem(bool verbose);
void init_icosphere();

void fill_vertices();

void quickSort(void * arr, int low, int high, int partition_fun(void *, int, int));

void free_cpu_memory();


#endif // CUDA_HEADER_CUH_