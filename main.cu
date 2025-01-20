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

// Host function to perform bitonic sort with CUDA
void bitonicSortCUDA(int *h_array, int N) {

    // Allocate memory on the GPU
    int *d_array;
    cudaMalloc(&d_array, N * sizeof(int));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;
    splitArrayIntoThreads<<<blocksPerGrid, threadsPerBlock>>>(h_array, d_array, N);

    int step_count = log2(N + 1);
    int partner;

    for (int step = 1; step <= step_count; step++) {
        int dist = 1 << (step - 1);
        
        if (threadIdx.x % (2 * dist) < dist) {
            partner = threadIdx.x + dist;
        } else {
            partner = threadIdx.x - dist;
        }
        if (partner >= 0 && partner < threadsPerBlock){
            if ((threadIdx.x / dist) % 2 == 0) {
                if (d_array[threadIdx.x] > d_array[partner]) {
                    int temp = d_array[threadIdx.x];
                    d_array[threadIdx.x] = d_array[partner];
                    d_array[partner] = temp;
                }
            } else {
                if (d_array[threadIdx.x] < d_array[partner]) {
                    int temp = d_array[threadIdx.x];
                    d_array[threadIdx.x] = d_array[partner];
                    d_array[partner] = temp;
                }
            }
        }
        // Perform bitonic merge for this step
        bitonic_merge<<<blocksPerGrid, threadsPerBlock>>>(d_array, 0, N, 1);  // Ascending order
        cudaDeviceSynchronize();  // Ensure all threads finish before moving on
    }

    // Final pass: Perform bitonic merge iteratively for the entire array size
    int currentSize = 2;
    while (currentSize <= N) {
        bitonic_merge<<<blocksPerGrid, threadsPerBlock>>>(d_array, 0, currentSize, 1);  // Ascending order merge
        currentSize *= 2;  // Double the size of the merged section
        cudaDeviceSynchronize(); // Ensure proper synchronization
    }

    // Copy the sorted array back to the host
    cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the device memory
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
