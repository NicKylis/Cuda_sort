#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "array_gen.h"

// CUDA kernel for sorting processes within a block
__global__ void sortLocalCUDA(int rank, int num_q, int *array) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num_q) {
        if (rank % 2 == 0) {
            // Ascending sort (Even processes)
            for (int i = 0; i < num_q - 1; i++) {
                for (int j = i + 1; j < num_q; j++) {
                    if (array[i] > array[j]) {
                        int temp = array[i];
                        array[i] = array[j];
                        array[j] = temp;
                    }
                }
            }
        } else {
            // Descending sort (Odd processes)
            for (int i = 0; i < num_q - 1; i++) {
                for (int j = i + 1; j < num_q; j++) {
                    if (array[i] < array[j]) {
                        int temp = array[i];
                        array[i] = array[j];
                        array[j] = temp;
                    }
                }
            }
        }
    }
}

// CUDA kernel for minmax operation between threads
__global__ void minmaxCUDA(int rank, int partner_rank, int num_q, int *array,
                            bool sort_descending) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ int shared_array[1024];  // 1024 threads per block

    if (idx < num_q) {
        shared_array[idx] = array[idx];
    }
    __syncthreads();

    if ((!sort_descending && rank < partner_rank) || (sort_descending && rank > partner_rank)) {
        // Keep min / max depending on the thread rank
        for (int i = 0; i < num_q; i++) {
            if (array[i] > shared_array[i]) {
                array[i] = shared_array[i];
            }
        }
    } else {
        for (int i = 0; i < num_q; i++) {
            if (array[i] < shared_array[i]) {
                array[i] = shared_array[i];
            }
        }
    }
}

// Main bitonic sort kernel
void bitonicSortCUDA(int rank, int num_p, int num_q, int *array) {
    // Sorting the processes (threads in blocks)
    int *d_array;
    cudaMalloc((void **)&d_array, num_q * sizeof(int));
    cudaMemcpy(d_array, array, num_q * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1024;  // Number of threads per block
    int numBlocks = (num_q + blockSize - 1) / blockSize;

    // Sorting within threads
    sortLocalCUDA<<<numBlocks, blockSize>>>(rank, num_q, d_array);
    cudaDeviceSynchronize();

    for (int group_size = 2; group_size <= num_p; group_size *= 2) {
        bool sort_descending = rank & group_size;
        for (int distance = group_size / 2; distance > 0; distance /= 2) {
            int partner_rank = rank ^ distance;

            // Perform minmax operation between partner threads
            minmaxCUDA<<<numBlocks, blockSize>>>(rank, partner_rank, num_q, d_array, sort_descending);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(array, d_array, num_q * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}