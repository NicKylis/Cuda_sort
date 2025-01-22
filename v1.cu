#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "array_gen.h"

// Bitonic sort kernel for local sorting using shared memory
__global__ void bitonicSortLocalCUDA(int N, int *array) {
    extern __shared__ int shared_array[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;

    // Load data into shared memory
    if (idx < N) {
        shared_array[thread_id] = array[idx];
    } else {
        shared_array[thread_id] = INT_MAX;
    }
    __syncthreads();

    // Bitonic sort within the block
    for (int k = 2; k <= blockDim.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = thread_id ^ j;
            if (ixj > thread_id) {
                if ((thread_id & k) == 0) {
                    if (shared_array[thread_id] > shared_array[ixj]) {
                        int temp = shared_array[thread_id];
                        shared_array[thread_id] = shared_array[ixj];
                        shared_array[ixj] = temp;
                    }
                } else {
                    if (shared_array[thread_id] < shared_array[ixj]) {
                        int temp = shared_array[thread_id];
                        shared_array[thread_id] = shared_array[ixj];
                        shared_array[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write back sorted data to global memory
    if (idx < N) {
        array[idx] = shared_array[thread_id];
    }
}

__global__ void mergeGlobalCUDA(int *array, int N, int subarray_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread merges two subarrays of size `subarray_size`
    int start = (tid * 2 * subarray_size);
    int mid = start + subarray_size;
    int end = min(start + 2 * subarray_size, N);

    if (start >= N || mid >= N) return; // Bounds check

    // Temporary buffer for merging
    int *temp = new int[end - start];

    int left = start;
    int right = mid;
    int out_idx = 0;

    // Merge two sorted subarrays into `temp`
    while (left < mid && right < end) {
        if (array[left] <= array[right]) {
            temp[out_idx++] = array[left++];
        } else {
            temp[out_idx++] = array[right++];
        }
    }

    // Copy remaining elements
    while (left < mid) {
        temp[out_idx++] = array[left++];
    }
    while (right < end) {
        temp[out_idx++] = array[right++];
    }

    // Write back the sorted subarray to the original array
    for (int i = 0; i < (end - start); i++) {
        array[start + i] = temp[i];
    }

    delete[] temp; // Free temporary buffer
}

void multiBlockSortGlobalCUDA(int N, int *array) {
    int *d_array;
    cudaMalloc((void **)&d_array, N * sizeof(int));
    cudaMemcpy(d_array, array, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1024;  // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Step 1: Sort each block using bitonic sort
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bitonicSortLocalCUDA<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(N, d_array);
    cudaDeviceSynchronize();

    // Step 2: Iteratively merge sorted blocks
    int subarray_size = blockSize;
    while (subarray_size < N) {
        int num_merge_blocks = (N + 2 * subarray_size - 1) / (2 * subarray_size);
        mergeGlobalCUDA<<<num_merge_blocks, blockSize>>>(d_array, N, subarray_size);
        cudaDeviceSynchronize();
        subarray_size *= 2;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}