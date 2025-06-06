#include <stdio.h>

int main(){
    int x = 1;
    float y = 1.23;
    void* vptr;

    vptr = &x;
    printf("%d\n", *(int*)vptr);
    // vptr is a memory address "&x" but it is stored as a void pointer (no data type)
    // We can't dereference a void pointer, so we cast it to an integer pointer to store the integer value at that memory address "(int*)vptr"
    // Then we dereference it with the final asterisk "*" to get the value "*((int*)vptr)"

    vptr = &y;
    printf("%f\n", *(float*)vptr);

    return 0;
}