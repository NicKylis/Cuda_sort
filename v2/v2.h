#ifndef V2_H
#define V2_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void bitonicSortLocalCUDA(int N, int *array);

void mergeCPU(int *array, int start, int mid, int end);

void multiwayMergeCPU(int N, int *array, int chunk_size);

void v2_sort(int N, int *array);

#endif