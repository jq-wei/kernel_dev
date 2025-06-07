#include <stdio.h>

// examples for each conditional macro
// #if
// #ifdef
// #ifndef  if not define
// #elif
// #else
// #endif

#define PI 3.1415
#define AREA(r) (PI*r*r) // lambda func

#ifndef radius
#define radius 7
#endif

#if radius>10
#define radius 10
#elif radius < 5
#define radius 5
#else
#define radius 7
#endif

int main(){
    printf("Area of a circle with radius %d: %f\n", radius, AREA(radius)); // Area of a circle with radius 7: 153.933500
}