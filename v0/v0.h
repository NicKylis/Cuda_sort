#ifndef V0_H
#define V0_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sortLocalCUDA(int N, int *array);

__global__ void sortLocal2CUDA(int rank, int num_q, int *array);

__global__ void mergeCUDA(int rank, int partner_rank, int num_q, int *array,
                            bool sort_descending);

void bitonicSortCUDA(int rank, int num_p, int num_q, int *array);

__global__ void mergeGlobalCUDA(int *array, int N, int subarray_size);

void v0_sort(int N, int *array);

#endif