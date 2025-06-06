#include <stdio.h>

int main() {
    int age = 18;
    int* ptr = &age; //ptr must be the same type of the var its points to.

    printf("%d\n", age);

    printf("%p\n", ptr);

    printf("%p\n", &age);

    printf("%d\n", *ptr); // dereference the value of a pointer

    // pointer of an array

    int num[4] = {10,20,30,40}; // !! the name of an array, is a POINTER to the first element of the array.

    printf("%p\n", num);
    printf("%p\n", &num[0]);  // these two printf gives the same mem address.

    // Dereference
    printf("the first value of arr %d\n", *num); // the first var of the arr
    printf("the second value of arr %d\n", *(num+1)); // the sec var of the arr
    printf("the 3rd value of arr %d\n", *(num+2)); // the 3rd var of the arr
    
    *num = 100; // change the value of the 1st element
    *(num+1) = 200; // change the value of the 2nd element

    printf("the first value of arr %d\n", *num); // the first var of the arr
    printf("the second value of arr %d\n", *(num+1)); // the sec var of the arr    

    return 0;
}