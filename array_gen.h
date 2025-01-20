#ifndef ARRAY_GEN_H
#define ARRAY_GEN_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MAX_NUM 1000 //Change this to the desired max value

void generate_random_array(int *arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % MAX_NUM;
    }
}

#endif