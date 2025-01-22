#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "array_gen.h"

// Sort local sections of the array within a block using shared memory
__global__ void sortLocalCUDA(int N, int *array) {
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

    // Perform local sorting in shared memory
    for (int i = 0; i < blockDim.x - 1; i++) {
        for (int j = i + 1; j < blockDim.x; j++) {
            if (shared_array[i] > shared_array[j]) {
                int temp = shared_array[i];
                shared_array[i] = shared_array[j];
                shared_array[j] = temp;
            }
        }
    }
    __syncthreads();

    // Write back sorted data to global memory
    if (idx < N) {
        array[idx] = shared_array[thread_id];
    }
}

__global__ void sortLocal2CUDA(int rank, int num_q, int *array) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num_q) {
        if (rank % 2 == 0) {
            // Ascending (Even thread ranks)
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
            // Descending (Odd thread ranks)
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
    sortLocal2CUDA<<<numBlocks, blockSize>>>(rank, num_q, d_array);
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

void v0_sort(int N, int *array) {
    int *d_array;
    cudaMalloc((void **)&d_array, N * sizeof(int));
    cudaMemcpy(d_array, array, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1024;  // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Step 1: Sort each block locally
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sortLocalCUDA<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(N, d_array);
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