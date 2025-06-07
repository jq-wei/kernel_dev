#include <stdio.h>

// no input var needed by setting (void)
__global__ void whoami(void) {
    int block_id = 
        blockIdx.x + 
        blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;

    int block_offset = 
        block_id * // times our apartment number
        blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment)

    int thread_offset =
        threadIdx.x +  
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset; // global person id in the entire apartment complex

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);

}


int main(int argc, char **argv) {
    const int b_x = 2, b_y = 3, b_z = 4;
    const int t_x = 4, t_y = 4, t_z = 4; // the max warp size is 32, so 
    // we will get 2 warp of 32 threads per block

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z); // 3d cube of shape 2*3*4 = 24, a grid with (2,3,4) blocks
    dim3 threadsPerBlock(t_x, t_y, t_z); // 3d cube of shape 4*4*4 = 64, a block with (4,4,4) threads

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}