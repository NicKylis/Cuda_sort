#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "array_gen.h"

// Sort local sections of the array within a block
__global__ void sortLocalCUDA(int num_q, int *array) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int block_start = blockIdx.x * num_q;
    int block_end = block_start + num_q;

    for (int i = block_start; i < block_end - 1; i++) {
        for (int j = i + 1; j < block_end; j++) {
            if (array[i] > array[j]) {
                int temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }
    }
}

// Merge sorted sections across multiple blocks
__global__ void mergeBlocksCUDA(int num_q, int num_blocks, int *array) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Global start and end indices for merging
    int start = 0;
    int end = num_blocks * num_q;

    for (int i = start; i < end - 1; i++) {
        for (int j = i + 1; j < end; j++) {
            if (array[i] > array[j]) {
                int temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }
    }
}

// Main function to perform sorting
void multiBlockSortCUDA(int N, int num_q, int *array) {
    int *d_array;
    cudaMalloc((void **)&d_array, N * sizeof(int));
    cudaMemcpy(d_array, array, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlocks = (N / num_q);

    // Step 1: Sort local blocks
    sortLocalCUDA<<<numBlocks, blockSize>>>(num_q, d_array);
    cudaDeviceSynchronize();

    // Step 2: Merge sorted blocks
    mergeBlocksCUDA<<<1, blockSize>>>(num_q, numBlocks, d_array);
    cudaDeviceSynchronize();

    // Copy back the result
    cudaMemcpy(array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <log2(size)>\n", argv[0]);
        return -1;
    }

    int q = atoi(argv[1]);
    const int N = 1 << q;
    const int num_q = 1024;  // Size per block

    // Generate a random array
    int *h_array = (int *)malloc(N * sizeof(int));
    generate_random_array(h_array, N);

    // Perform the multi-block sort using CUDA
    multiBlockSortCUDA(N, num_q, h_array);

    // Print the sorted array
    printf("Sorted Array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    free(h_array);
    return 0;
}
