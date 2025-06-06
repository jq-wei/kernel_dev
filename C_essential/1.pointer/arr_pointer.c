#include <stdio.h>
//matrix

int main(){
    int arr1[] = {1,2,3,4};
    int arr2[] = {5,6,7,8};

    int* ptr1 = arr1;
    int* ptr2 = arr2;

    int* matrix1[] = {ptr1, ptr2};
    int* matrix2[] = {arr1, arr2};

    for (int i = 0; i<2; i++){
        for (int j=0; j<4; j++){
            printf("%d ", *matrix1[i]++);
        }
        printf("\n");
    }


    for (int i = 0; i<2; i++){
        for (int j=0; j<4; j++){
            printf("%d ", *matrix2[i]++);
        }
        printf("\n");
    }
}