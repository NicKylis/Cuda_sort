#ifndef V1_H
#define V1_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void bitonicSortLocalCUDA(int N, int *array);

__global__ void mergeGlobalCUDA(int *array, int N, int subarray_size);

void v1_sort(int N, int *array);

#endif