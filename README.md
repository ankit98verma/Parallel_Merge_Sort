# Parallel merge sort

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This repository implements the parallel merge sort algorithm on CUDA. The following article explains the parallel merge sorting of two arrays assuming that no duplicate elements are present in the arrays. 
[ICS 643: Advanced Parallel Algorithms, Prof:Nodari Sitchinava](http://www2.hawaii.edu/~nodari/teaching/f16/notes/notes10.pdf)

The above algorithm has been modified to remove the constrain of no duplicate elements in the arrays.
## Modified parallel merge sort algorithm
Consider two integer arrays i.e. *arr1* and *arr2*. We assume, without the loss of generality, that the size of both the arrays is "length". The merged result is written to array *res_arr*. The algorithm is:

1. (In parallel) For every element in arr1 perform the following:
    a.  Find the index of largest number in *arr2* which is smaller than arr1[i].

            ![formula](http://www.sciweavers.org/tex2img.php?eq=lsIndex[i]=argmax_j%20(arr2[j]%20%3C%20arr1[i])&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)

    b. The *lsIndex[i]* is the index of largest number in *arr2* array which is smaller than *arr1[i]*. Now put the *arr1[i]* into the *res_arr* at position *lsIndex[i]+i+1*.
2. (In parallel) For every element in arr2 perform the following:
    a.  Find the index of smallest number in *arr1* which is larger than arr2[i].

            ![formula](http://www.sciweavers.org/tex2img.php?eq=slIndex[i]=argmin_j%20(arr2[i]%20%3C%20arr1[j])&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)
            
    b. The *slIndex[i]* is the index of smallest number in *arr1* array which is larger than *arr1[i]*. Now put the *arr2[i]* into the *res_arr* at position *slIndex[i]+i*.
## Implementation
The parallel merging implemented in the "cuda_sortin.cu" file is as follows:
1. First merge the arrays in sequential manner in CUDA till the resulting array's size is less than equal to 1024.
    a. The array is divided into multiple parts.
    b. Each thread merges two parts of the array, i.e. each thread works on two arrays.
2. Now merge all the chunks of size 1024 using the parallel merging algorithm.
    a. 