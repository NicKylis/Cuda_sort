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

__global__ void mergeCUDA(int N, int *array, int subarray_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate start, mid, and end indices for this merge
    int start = (idx / subarray_size) * subarray_size * 2;
    int mid = start + subarray_size;
    int end = min(start + 2 * subarray_size, N);

    if (start >= N || mid >= N) return;

    // Temporary storage for merging
    extern __shared__ int temp[];

    int left = start;
    int right = mid;
    int out_idx = start;

    // Merge two sorted subarrays
    while (left < mid && right < end) {
        if (array[left] <= array[right]) {
            temp[out_idx - start] = array[left++];
        } else {
            temp[out_idx - start] = array[right++];
        }
        out_idx++;
    }

    // Copy remaining elements
    while (left < mid) {
        temp[out_idx - start] = array[left++];
        out_idx++;
    }
    while (right < end) {
        temp[out_idx - start] = array[right++];
        out_idx++;
    }

    // Write back merged subarray to global memory
    for (int i = start; i < end; i++) {
        array[i] = temp[i - start];
    }
}

void multiBlockSortCUDA(int N, int *array) {
    int *d_array;
    cudaMalloc((void **)&d_array, N * sizeof(int));
    cudaMemcpy(d_array, array, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1024;  // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Step 1: Sort each block locally
    sortLocalCUDA<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(N, d_array);
    cudaDeviceSynchronize();

    // Step 2: Iteratively merge sorted blocks
    int subarray_size = blockSize;
    while (subarray_size < N) {
        int num_merge_blocks = (N + 2 * subarray_size - 1) / (2 * subarray_size);
        mergeCUDA<<<num_merge_blocks, blockSize, 2 * subarray_size * sizeof(int)>>>(N, d_array, subarray_size);
        cudaDeviceSynchronize();
        subarray_size *= 2;  // Double the size of subarrays in the next step
    }

    // Copy result back to host
    cudaMemcpy(array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);
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

void multiBlockSortGlobalCUDA(int N, int *array) {
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

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start);
    multiBlockSortGlobalCUDA(N, h_array);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);

    // float milliseconds;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Print the sorted array
    // printf("Sorted Array: ");
    // for (int i = 0; i < N; i++) {
    //     printf("%d ", h_array[i]);
    // }
    // printf("\n");
    // printf("Time: %f ms\n", milliseconds);

    free(h_array);
    return 0;
}

