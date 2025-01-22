#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "v0.h"

#define MAX_NUM 1000 //Change this to the desired max value

void generate_random_array(int *arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % MAX_NUM;
    }
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
    v0_sort(N, h_array);

    // Uncomment this to print the sorted array
    // printf("Sorted Array: ");
    // for (int i = 0; i < N; i++) {
    //     printf("%d ", h_array[i]);
    // }
    printf("\n");

    free(h_array);

    return 0;
}
