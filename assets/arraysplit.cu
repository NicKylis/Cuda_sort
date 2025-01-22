#include <stdio.h>
#include <cuda_runtime.h>
#include "array_gen.h"
#include "sortnmerge.h"

__global__ void hello_from_gpu() {
    printf("Hello from GPU!\n");
}

__global__ void splitArrayIntoThreads(int* input, int* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N / 2) {
        output[2 * idx] = input[2 * idx];       // First integer for the thread
        output[2 * idx + 1] = input[2 * idx + 1];   // Second integer for the thread
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <log2(size)>\n", argv[0]);
        return -1;
    }

    int q = atoi(argv[1]);
    const int N = 1 << q;

    // Allocate memory for input and output arrays on the host (heap allocation)
    int *h_input = (int*)malloc(N * sizeof(int));
    int *h_output = (int*)malloc(N * sizeof(int));

    if (h_input == NULL || h_output == NULL) {
        printf("Failed to allocate host memory\n");
        return -1;
    }

    generate_random_array(h_input, N);

    // Print input array
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");

    int *d_input, *d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;
    splitArrayIntoThreads<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Copy output data back to host
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Thread outputs:\n");
    for (int i = 0; i < N / 2; i++) {
        printf("Thread %d: %d, %d\n", i, h_output[2 * i], h_output[2 * i + 1]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
//nvcc -o program program.cu