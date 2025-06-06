#include <iostream>

using namespace std;

typedef struct{
    float x;
    float y;
} Point;

int main(){
    Point p = {1.1, 2.5};
    printf("size of Point: %zu\n", sizeof(Point)); //8 bytes, 4*2

    float a = 13333.1;
    printf("size of float: %zu\n", sizeof(a)); //4 bytes.
}