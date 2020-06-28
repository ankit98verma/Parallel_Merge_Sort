# Parallel merge sort

This repository implements the parallel merge sort algorithm on CUDA. The following article explains the parallel merge sorting of two arrays assuming that no duplicate elements are present in the arrays. 
[ICS 643: Advanced Parallel Algorithms, Prof:Nodari Sitchinava](http://www2.hawaii.edu/~nodari/teaching/f16/notes/notes10.pdf)

The above algorithm has been modified to remove the constrain of no duplicate elements in the arrays.
## Modified parallel merge sort algorithm
Consider two integer arrays i.e. *arr1* and *arr2*. We assume, without the loss of generality, that the size of both the arrays is "length". The merged result is written to array *res_arr*. The algorithm is:

1. (In parallel) For every element in arr1 perform the following:

    a.  Find the index of largest number in *arr2* which is smaller than arr1[i].

	lsIndex[i] = argmax<sub>j</sub> (arr2[j] < arr1[i]).

    b. The *lsIndex[i]* is the index of largest number in *arr2* array which is smaller than *arr1[i]*. Now put the *arr1[i]* into the *res_arr* at position *lsIndex[i]+i+1*.

2. (In parallel) For every element in arr2 perform the following:

    a.  Find the index of smallest number in *arr1* which is larger than arr2[i].

    slIndex[i] = argmax<sub>j</sub> (arr2[i] < arr1[j]).

    b. The *slIndex[i]* is the index of smallest number in *arr1* array which is larger than *arr1[i]*. Now put the *arr2[i]* into the *res_arr* at position *slIndex[i]+i*.

## Implementation
The parallel merging implemented in the "cuda_sortin.cu" file is as follows:
1. First merge the arrays in sequential manner in CUDA till the resulting array's size is less than equal to 1024.
    
    - The array is divided into multiple parts.
    
    - Each thread merges two parts of the array, i.e. each thread works on two arrays.
2. Now merge all the chunks of size 1024 using the parallel merging algorithm.
    
    - In this each threads works on an element of the array and places the element at the proper position in the resulting array.

## Time complexity
Say, we have an array of size *N* to sort. There are total of *O(log<sub>2</sub>(N))* steps to sort the array using the merge sort. In each step we use binary search to find the largest smallest element and the smallest larget element for which the complexity is *O(log<sub>2</sub>(N))*. Hence the time complexity of the algorithm is *O(log<sup>2</sup>(N))* (the base of the log is 2).

## Building and running the example code
The *main.cpp* implements a example usage of the parallel merging. The code generates fills an array with random numbers between 0 and 100 uniformly, then sort the array and export is as ".csv" file in the "results" folder. The Matlab code provided in the "utility" folder can be used to plot the exported ".csv" file to verify the results.

Following steps are to be followed to build and run the program
1. Make sure we are in the same correct folder
    ```sh
    $ ls
    bin  Makefile  README.md  results  src  utilities
    ```
2. Make the file. Note that it is assumed that the nvidia cuda toolkit is installed (ignore warnings, if any).
    ```sh
    $ make clean all
    rm -f sort *.o bin/*.o *~
    rm -f src/*~
    g++ -g -Wall -D_REENTRANT -std=c++0x -pthread -c -o bin/gpu-main.cpp.o -I/include src/main.cpp 
    /bin/nvcc -m64 -g -dc -Wno-deprecated-gpu-targets --std=c++11 --expt-relaxed-constexpr -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61 -c -o bin/cuda_sorting.cu.o  src/cuda_sorting.cu
    /bin/nvcc -dlink -Wno-deprecated-gpu-targets -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61 -o bin/cuda.o  bin/cuda_sorting.cu.o
    g++ -g -Wall -D_REENTRANT -std=c++0x -pthread -o sort -I/include bin/gpu-main.cpp.o bin/cuda.o bin/cuda_sorting.cu.o -L/lib64 -lcudart -lcurand 
    ```
3. A new file "sort" is created, check using "ls"
    ```sh
    $ ls
    bin  Makefile  README.md  results  sort  src  utilities
    ```
4. Run the program using ./short <length of array> <verbose: 0/1>
    ```sh
    $ ./sort 100 1
    Verbose ON

    ----------Running GPU Code----------
    
    No kernel error detected
    GPU Input data copy time: 0.117760 ms
    GPU Sorting time: 0.070656 ms
    GPU Output data copy time: 0.015392 ms
    Total GPU time: 0.203808 ms
    Exporting: gpu_arr.csv
    -----------------------
    ```

## Using the parallel merge
The *cuda_sort.cu* and *cuda_sort.cuh* files are required to use the parallel merge sorting. Following code gives an example on how to use it:

```C++
#include <cstdio>
#include <cstdlib>
#include <iostream>

/* Cuda includes */
#include <cuda_runtime.h>
#include "cuda_sorting.cuh"

int main(int argc, char **argv) {

    /* allocate the memory for the array to sorted and the resulting array */
    int arr_len = 100;
    cpu_arr = (int *)malloc(arr_len*sizeof(float));
    gpu_out_arr = (int *)malloc(arr_len*sizeof(int));
    
    /* Initialize the array (here it is randomly initialized) */
    srand(0);
    for(unsigned int i = 0; i<arr_len; i++){
        cpu_arr[i] = rand()%100;
    }
    
    /* Initialize the GPU memory and copy the array to it */
    cuda_cpy_input_data(cpu_arr, arr_len);
    
    /* sort the array in GPU */
    cudacall_merge_sort();
    /* check for errors */
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err){
        cerr << "Error " << cudaGetErrorString(err) << endl;
    }else{
    	cerr << "No kernel error detected" << endl;
    }
    /* Copy the result back to the CPU memory*/
    cuda_cpy_output_data(gpu_out_arr, arr_len);
    
    /* Free the CPU memory */
    free(cpu_arr);
	free(gpu_out_arr);
	
	/* Free the GPU memory */
	free_gpu_memory();
	
    return 1;
}
```
