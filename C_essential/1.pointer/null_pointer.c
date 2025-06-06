// Purpose: Demonstrate NULL pointer initialization and safe usage.

// Key points:
// 1. Initialize pointers to NULL when they don't yet point to valid data.
// 2. Check pointers for NULL before using to avoid crashes.
// 3. NULL checks allow graceful handling of uninitialized or failed allocations.

#include <stdio.h>
#include <stdlib.h>


int main(){
    // Initialize a null pointer
    int* ptr = NULL;
    printf("1. initial ptr value %p\n", ptr); //or  printf("1. initial ptr value %p\n", (void*)ptr); 

    // check for NULL before using a ptr
    if (ptr == NULL){
        printf("2. ptr is NULL, cannot dereference\n");
    }

    // Allocate mem to a ptr
    ptr = malloc(sizeof(int));
    if (ptr == NULL){
        printf("3. mem allocation failed.\n");
    } else {
        printf("3. ptr after malloc %p\n", ptr);
    }

    printf("4. After allocation, ptr value: %p\n", (void*)ptr);
    printf("4+. After allocation, ptr value: %i\n", *(int*)ptr);
    printf("Size of ptr: %zu bytes\n", sizeof(ptr));  // 8. The output Size of ptr: 8 bytes indicates the size of the pointer itself, not the memory it points to, which should be 4. 

    // Safe to use ptr after NULL check
    *ptr = 42;
    printf("5. Value at ptr: %d\n", *ptr);

    // Clean up
    free(ptr);
    ptr = NULL;  // Set to NULL after freeing

    printf("6. After free, ptr value: %p\n", (void*)ptr);

    // Demonstrate safety of NULL check after free
    if (ptr == NULL) {
        printf("7. ptr is NULL, safely avoided use after free\n");
    }

    return 0;
}
