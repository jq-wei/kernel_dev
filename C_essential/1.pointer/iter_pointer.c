#include <stdio.h>

int main(){
    int arr[] = {12,13,14,15};
    int* ptr = arr; //ptr to the 1st element

    for (int i=0; i<4;i++) {
        printf("%d\n", *ptr);
        printf("%p\n", ptr);
        ptr++;
    }

    //12
    //0x7ffc17d03270
    //13
    //0x7ffc17d03274
    //14
    //0x7ffc17d03278
    //15
    //0x7ffc17d0327c

    // notice that the pointer is incremented by 4 bytes (size of int = 4 bytes * 8 bits/bytes = 32 bits = int32) each time. 
    // 这是因为 ptr is of type int*, which takes 4 bytes.
    // ptrs are 64 bits in size (8 bytes). 2**32 = 4,294,967,296 which is too small given how much memory we typically have.
    // 这里想说，ptrs 自己是 64 bits （8 bytes）. 如果 ptrs 是32 bits，那只能有大概4GB  （4,294,967,296） 大小，对于长度大于4GB的序列，就会不够用。
    // arrays are layed out in memory in a contiguous manner (one after the other rather than at random locations in the memory grid)
}