#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "array_gen.h"
#include "v0.h"
#include "v1.h"
#include "v2.h"

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
    v0_sort(0, N / 2, N, h_array);

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
