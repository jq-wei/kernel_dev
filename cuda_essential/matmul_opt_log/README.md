# this is the log of matmul optimization 

From naive to cuBLAS level. This is learning note of [blog](https://siboehm.com/)

Here given A (M*K), B (K *N), C ( M *N), and compute `alpha*(A*B)+beta *C`. 