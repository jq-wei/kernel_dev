#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256  // Number of rows in A and C
#define K 512   // Number of columns in A and rows in B
#define N 256  // Number of columns in B and C
#define alpha 0.5f
#define beta 0.7f
#define BLOCK_SIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b)) 

void sgemm_naive_cpu(int m, int n, int k, float alpha_v, const float *A,
                    const float *B, float beta_v, float *C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = alpha_v * sum + beta_v * C[i * n + j];
        }
    }
}


__global__ void sgemm_naive(int m, int n, int k, float alpha_v, const float *A,
                            const float *B, float beta_v, float *C) {
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.y * blockDim.y + threadIdx.y;
    const uint y = blockIdx.x * blockDim.x + threadIdx.x;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < m && y < n) {
        float tmp = 0.0f;
        for (int i = 0; i < k; i++) {
          tmp += A[x * k + i] * B[i * n + y];
        }
        // C = α*(A@B)+β*C
        C[x * n + y] = alpha_v * tmp + beta_v * C[x * n + y];
    }
}

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    float *h_A, *h_B, *h_C, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size_A); //malloc return a void pointer, then casted to (float*)
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);
    
    // Initialize matrices
    srand(time(NULL)); // seed random. Without seeding, rand() produces the same sequence every time.
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    init_matrix(h_C, M, N);

    // Allocate device memory
    cudaMalloc(&d_A, size_A); // pointer of pointer: d_A is a pointer to A in device, &d_A is the address of d_A. 
    cudaMalloc(&d_B, size_B); // cuda allocates memory on the GPU and updates d_A to point to that memory. 
    cudaMalloc(&d_C, size_C); // It changes d_A by using a pointer to it.

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(M,32), CEIL_DIV(N,32), 1);


    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_milliseconds = 0.0f;
    

    // Here d_c is modified, so only one run to match cpu res. If matching is not 
    // important, we can run more iterations to have averaged profile.
    for (int i = 0; i < 1; i++) {
        
        cudaEventRecord(start);

        sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        // Stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // Wait for kernel completion
        
        // Get elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_milliseconds += milliseconds;
    }
    // Calculate average
    float avg_ms = total_milliseconds / 20;
    float avg_seconds = avg_ms / 1000.0f;

    // Compute GFLOPs (example for M=256, K=512, N=256)
    float gflops = (256*256*2*512 / avg_seconds) / 1e9;

    printf("Average Kernel Time: %.3f ms\n", avg_ms);
    printf("Average Performance: %.2f GFLOPs/s\n", gflops);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // compare result with cpu
    float *h_C_ref = (float*)malloc(size_C);
    memcpy(h_C_ref, h_C, size_C);
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    // compute cpu res
    sgemm_naive_cpu(M,N,K,alpha, h_A, h_B, beta, h_C_ref);

    // compare res
    float max_abs_diff = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_C_ref[i] - h_C_gpu[i]);
        if (diff > 1e-3f) {
            if (mismatches < 10) {
                printf("Mismatch at index %d: CPU=%f, GPU=%f, diff=%f\n", i, h_C[i], h_C_gpu[i], diff);
            }
            mismatches++;
        }
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    if (mismatches == 0) {
        printf("✅ GPU result matches CPU result within tolerance.\n");
    } else {
        printf("❌ %d mismatches found. Max absolute difference: %f\n", mismatches, max_abs_diff);
    }

    // Free host and device memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}