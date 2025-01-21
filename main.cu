#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "array_gen.h"

__global__ void splitArrayIntoThreads(int* input, int* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N / 2) {
        output[2 * idx] = input[2 * idx];       // First integer for the thread
        output[2 * idx + 1] = input[2 * idx + 1];   // Second integer for the thread
    }
}

// CUDA kernel for performing bitonic merge
__global__ void bitonic_merge(int *d_array, int low, int cnt, int dir) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread processes two elements, comparing and swapping them
    int i = low + idx * 2;
    int j = i + cnt / 2;

    // Make sure indices are within bounds
    if (i < low + cnt - cnt / 2 && j < low + cnt) {
        if ((d_array[i] > d_array[j]) == dir) {
            // Swap elements if needed
            int temp = d_array[i];
            d_array[i] = d_array[j];
            d_array[j] = temp;
        }
    }
}

// CUDA kernel for performing bitonic sort
__global__ void bitonic_sort(int *d_array, int low, int cnt, int dir) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread processes two elements
    int i = low + idx * 2;
    int j = i + cnt / 2;

    // Make sure indices are within bounds
    if (i < low + cnt - cnt / 2 && j < low + cnt) {
        if ((d_array[i] > d_array[j]) == dir) {
            // Swap elements if needed
            int temp = d_array[i];
            d_array[i] = d_array[j];
            d_array[j] = temp;
        }
    }
}

void split(int *arr, int low, int cnt, int dir) {
    for (int size = 2; size <= cnt; size = size * 2) {
        for (int i = low; i < low + cnt - size; i++) {
            int j = i + size / 2;
            if ((arr[i] > arr[j]) == dir) {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
}

#include <cuda.h>
#include <cmath>
#include <cstdio>

__global__ void bitonic_sort_step(int* d_array, int N, int step, int dist) {
    extern __shared__ int shared_array[];

    int thread_idx = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (global_idx < N) {
        shared_array[thread_idx] = d_array[global_idx];
    }
    __syncthreads();

    int partner;
    if (thread_idx % (2 * dist) < dist) {
        partner = thread_idx + dist;
    } else {
        partner = thread_idx - dist;
    }

    // Ensure partner is within bounds of shared memory
    if (partner >= 0 && partner < blockDim.x && global_idx < N) {
        if ((thread_idx / dist) % 2 == 0) {
            if (shared_array[thread_idx] > shared_array[partner]) {
                // Swap values in shared memory
                int temp = shared_array[thread_idx];
                shared_array[thread_idx] = shared_array[partner];
                shared_array[partner] = temp;
            }
        } else {
            if (shared_array[thread_idx] < shared_array[partner]) {
                // Swap values in shared memory
                int temp = shared_array[thread_idx];
                shared_array[thread_idx] = shared_array[partner];
                shared_array[partner] = temp;
            }
        }
    }
    __syncthreads();

    // Write shared memory back to global memory
    if (global_idx < N) {
        d_array[global_idx] = shared_array[thread_idx];
    }
}

void bitonicSortCUDA(int* h_array, int N) {
    // Allocate memory on the GPU
    int* d_array;
    cudaMalloc(&d_array, N * sizeof(int));
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    int step_count = log2(N);
    for (int step = 1; step <= step_count; step++) {
        int dist = 1 << (step - 1);

        // Launch kernel for each step
        size_t sharedMemSize = threadsPerBlock * sizeof(int); // Shared memory size
        bitonic_sort_step<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, N, step, dist);
        cudaDeviceSynchronize(); // Ensure all steps complete before moving to the next
    }

    // Copy the sorted array back to the host
    cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_array);
}


int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <log2(size)>\n", argv[0]);
        return -1;
    }

    int q = atoi(argv[1]);
    const int N = 1 << q;

    // Generate a random array
    int *h_array = (int *)malloc(N * sizeof(int));
    generate_random_array(h_array, N);

    // Perform the bitonic sort using CUDA
    bitonicSortCUDA(h_array, N);

    // Print the sorted array
    printf("Sorted Array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Free the allocated memory
    free(h_array);

    return 0;
}
