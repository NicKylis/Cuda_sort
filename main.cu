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

// CUDA kernel for sorting processes within a block
__global__ void sortProcessesCUDA(int rank, int num_q, int *array) {
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

    __shared__ int shared_array[256];  // Assume 256 threads per block

    if (idx < num_q) {
        shared_array[idx] = array[idx];
    }
    __syncthreads();

    if ((!sort_descending && rank < partner_rank) || (sort_descending && rank > partner_rank)) {
        // Keep min elements
        for (int i = 0; i < num_q; i++) {
            if (array[i] > shared_array[i]) {
                array[i] = shared_array[i];
            }
        }
    } else {
        // Keep max elements
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

    int blockSize = 256;  // Number of threads per block
    int numBlocks = (num_q + blockSize - 1) / blockSize;

    // Sorting within processes (threads)
    sortProcessesCUDA<<<numBlocks, blockSize>>>(rank, num_q, d_array);
    cudaDeviceSynchronize();

    for (int group_size = 2; group_size <= num_p; group_size *= 2) {
        bool sort_descending = rank & group_size;
        for (int distance = group_size / 2; distance > 0; distance /= 2) {
            int partner_rank = rank ^ distance;

            // Perform minmax operation between partner threads
            minmaxCUDA<<<numBlocks, blockSize>>>(rank, partner_rank, num_q, d_array, sort_descending);
            cudaDeviceSynchronize();
        }
        // Assuming elbowMerge function will also be parallelized in a similar manner
        // elbowMerge(num_p, num_q, array, sort_descending);
    }

    cudaMemcpy(array, d_array, num_q * sizeof(int), cudaMemcpyDeviceToHost);
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
    bitonicSortCUDA(0, N / 2, N, h_array);

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
