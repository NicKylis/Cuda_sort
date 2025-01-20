#include <stdio.h>
#include <cuda_runtime.h>
#include "array_gen.h"

__global__ void hello_from_gpu() {
    printf("Hello from GPU!\n");
}

__global__ void splitArrayIntoThreads(int* input, int* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N / 2) {
        output[2 * idx] = input[2 * idx];       // First integer for the thread
        output[2 * idx + 1] = input[2 * idx + 1];;   // Second integer for the thread
    }
}

int main() {

    const int N = 10;
    int h_input[N];
    generate_random_array(h_input, N); //Example input array

    // Print input array
    for(int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }   
    // Allocate memory for output arrays
    int h_output[N];

    int *d_input, *d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
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
    return 0;
}

//nvcc -o program program.cu