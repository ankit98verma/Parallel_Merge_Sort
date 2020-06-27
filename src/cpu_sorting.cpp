/*
 * grav_run: Ankit Verma, Garima Aggarwal, 2020
 *
 * This file contains the code for gravitational field calculation
 * by using CPU.
 *
 */

#ifndef _GRAV_CPU_C_
	#define _GRAV_CPU_C_
#endif
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>
#include <sstream>


#include "cpu_sorting.hpp"

using namespace std;


/*Reference for this file: http://www.songho.ca/opengl/gl_sphere.html*/

/* Decleration of local functions */
int partition_sum(float * arr, int low, int high);

/*******************************************************************************
 * Function:        init_vars
 *
 * Description:     This function initializes global variables. This should be
 *					the first function to be called from this file.
 *
 * Arguments:       unsigned int depth: The maximum depth of the icosphere
 *					float r: The radius of sphere
 *
 * Return Values:   None.
 *
*******************************************************************************/
void init_vars(unsigned int len){
	faces_length = len;
    cpu_arr = (float *)malloc(faces_length*sizeof(float));
    srand(0);
    for(unsigned int i = 0; i<faces_length; i++){
        cpu_arr[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        cout << cpu_arr[i] << endl;
    }
}


void sort_cpu_arr(){
	quickSort(cpu_arr, 0, faces_length-1, partition_sum);
    cout << "--------------------------------"<< endl;
    for(unsigned int i = 0; i<faces_length; i++){
        cout << cpu_arr[i] << endl;
    }
}

int partition_sum(float * arr, int low, int high){
    float pivot = arr[high]; // pivot
    int i = (low - 1); // Index of smaller element
  	float tmp;
    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++; // increment index of smaller element
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
    tmp = arr[high];
    arr[high] = arr[i+1];
    arr[i+1] = tmp;
    return (i + 1);
}

void quickSort(float * arr, int low, int high, int partition_fun(float *, int, int)){
	if(low < high){
		/* pi is partitioning index, arr[p] is now
	    at right place */
	    int pi = partition_fun(arr, low, high);

	    // Separately sort elements before
	    // partition and after partition
	    quickSort(arr, low, pi - 1, partition_fun);
	    quickSort(arr, pi + 1, high, partition_fun);
	}

}


void free_cpu_memory(){

    // Free malloc arrays
    free(cpu_arr);
}

