# Parallel merge sort

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This repository implements the parallel merge sort algorithm on CUDA. The following article explains the parallel merge sorting of two arrays assuming that no duplicate elements are present in the arrays. 
[ICS 643: Advanced Parallel Algorithms, Prof:Nodari Sitchinava](http://www2.hawaii.edu/~nodari/teaching/f16/notes/notes10.pdf)

The above algorithm has been modified to remove the constrain of no duplicate elements in the arrays.
## Modified parallel merge sort algorithm
Consider two integer arrays i.e. *arr1* and *arr2*. We assume, without the loss of generality, that the size of both the arrays is "length". The merged result is written to array *res_arr*. The algorithm is:

1. (In parallel) For every element in arr1 perform the following:

    a.  Find the index of largest number in *arr2* which is smaller than arr1[i].
![formula1](http://www.sciweavers.org/tex2img.php?eq=lsIndex[i]=argmax_j%20(arr2[j]%20%3C%20arr1[i])&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)

    b. The *lsIndex[i]* is the index of largest number in *arr2* array which is smaller than *arr1[i]*. Now put the *arr1[i]* into the *res_arr* at position *lsIndex[i]+i+1*.

2. (In parallel) For every element in arr2 perform the following:

    a.  Find the index of smallest number in *arr1* which is larger than arr2[i].
![formula2](http://www.sciweavers.org/tex2img.php?eq=slIndex[i]=argmin_j%20(arr2[i]%20%3C%20arr1[j])&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)

    b. The *slIndex[i]* is the index of smallest number in *arr1* array which is larger than *arr1[i]*. Now put the *arr2[i]* into the *res_arr* at position *slIndex[i]+i*.
    
## Implementation
The parallel merging implemented in the "cuda_sortin.cu" file is as follows:
1. First merge the arrays in sequential manner in CUDA till the resulting array's size is less than equal to 1024.
    
    - The array is divided into multiple parts.
    
    - Each thread merges two parts of the array, i.e. each thread works on two arrays.
    
2. Now merge all the chunks of size 1024 using the parallel merging algorithm.
    
    - In this each threads works on an element of the array and places the element at the proper position in the resulting array.

## Building and running the example program
The *main.cpp* implements a example usage of the parallel merging. The code generates fills an array with random numbers between 0 and 100 uniformly, then sort the array and export is as ".csv" file in the "results" folder. The Matlab code provided in the "utility" folder can be used to plot the exported ".csv" file to verify the results.

Following steps are to be followed to build and run the program
1. Make sure we are in the same correct folder
    ```sh
    $ ls
    bin  Makefile  README.md  results  src  utilities
    ```
2. Make the file. Note that it is assumed that the nvidia cuda toolkit is installed.
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