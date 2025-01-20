#ifndef SORTNMERGE_H
#define SORTNMERGE_H

void bitonic_merge(int *arr, int low, int cnt, int dir) {
    for (int size = 2; size <= cnt; size = size * 2) {
        for (int i = low; i < low + cnt - size; i++) {
            int j = i + size / 2;
            if ((arr[i] > arr[j]) == dir) {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        size *= 2;
    }
}

void bitonic_sort(int *arr, int low, int cnt, int dir) {
    for (int size = 2; size <= cnt; size *= 2) {
        for (int i = low; i < low + cnt - size; i++) {
            int j = i + size / 2;
            if ((arr[i] > arr[j]) == dir) {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        bitonic_merge(arr, low, cnt, dir);
    }
}

#endif