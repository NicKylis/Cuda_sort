#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Bitonic sort kernel for local sorting using shared memory
__global__ void bitonicSortLocalCUDA(int N, int *array) {
    extern __shared__ int shared_array[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;

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

// CPU function to merge two sorted arrays
void mergeCPU(int *array, int start, int mid, int end) {
    int *temp = new int[end - start];
    int left = start, right = mid, idx = 0;

    while (left < mid && right < end) {
        if (array[left] <= array[right]) {
            temp[idx++] = array[left++];
        } else {
            temp[idx++] = array[right++];
        }
    }

    while (left < mid) temp[idx++] = array[left++];
    while (right < end) temp[idx++] = array[right++];

    for (int i = 0; i < idx; i++) {
        array[start + i] = temp[i];
    }
    delete[] temp;
}

// CPU function for multiway merge
void multiwayMergeCPU(int N, int *array, int chunk_size) {
    while (chunk_size < N) {
        for (int start = 0; start < N; start += 2 * chunk_size) {
            int mid = start + chunk_size;
            int end = min(start + 2 * chunk_size, N);
            if (mid < end) {
                mergeCPU(array, start, mid, end);
            }
        }
        chunk_size *= 2;
    }
}

void v2_sort(int N, int *array) {
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

    // Copy the partially sorted array back to the host
    cudaMemcpy(array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    

    // Step 2: Perform the final merging on the CPU
    int chunk_size = blockSize;
    multiwayMergeCPU(N, array, chunk_size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);
}