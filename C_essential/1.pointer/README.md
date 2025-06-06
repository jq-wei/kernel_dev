A pointer is a variable that stores the mem address of another variable as its value. 
It points to a data type of the same type.

void pointers are used when we don't know the data type of the memory address.
fun fact: malloc() returns a void pointer but we see it as a pointer to a specific data type 
after the cast (int*)malloc(4) or (float*)malloc(4) etc.