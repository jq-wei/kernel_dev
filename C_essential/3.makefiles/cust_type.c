// size_t: size type for memory allocation; unsigned integer data type to represent the size of an 
// object in bytes (an unsigned long long int).

// It is guranteed to be big enough to contain the size of the largest object the host system can handle.

#include <stdio.h>
#include <stdlib.h>

int main(){
    int arr[] = {12,24,36,48};

    // size_t
    size_t size = sizeof(arr) / sizeof(arr[0]);
    size_t size1 = sizeof(arr) / sizeof(*arr);

    printf("size of arr: %zu\n", size); //4
    printf("size of arr: %zu\n", size1);//4

    printf("size of size_t: %zu\n", sizeof(size_t)); // 8 bytes -> 64 bits which is mem safe.
    printf("int size in bytes: %zu\n", sizeof(int)); // 4 bytes -> 32 bits

    // z -> size_t
    // u -> unsigned int
    // %zu -> size_t
    // src: https://cplusplus.com/reference/cstdio/printf/  

    return 0; 
}