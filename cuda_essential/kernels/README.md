All the benchmark is done with A5000.

# vec_add_v1

This version uses 1d block, which achieves 798x speed up comparing to cpu code. 

```bash
Performing warm-up runs...
Benchmarking CPU implementation...
Benchmarking GPU implementation...
CPU average time: 149.788566 milliseconds
GPU average time: 0.187682 milliseconds
Speedup: 798.098506x
Results are correct
```

# vec_add_v3 

this one use 3d grid, but each thread has more calculations, so slower.

```bash
Performing warm-up runs...
Benchmarking CPU implementation...
Benchmarking GPU 1D implementation...
1D Results are correct
Benchmarking GPU 3D implementation...
3D Results are correct
CPU average time: 159.361351 milliseconds
GPU 1D average time: 0.228311 milliseconds
GPU 3D average time: 0.247877 milliseconds
Speedup (CPU vs GPU 1D): 698.000708x
Speedup (CPU vs GPU 3D): 642.905971x
Speedup (GPU 1D vs GPU 3D): 0.921068x
```

# matmul_v1 
This is naive implementation of matmul. NO TILING.
```bash
Performing warm-up runs...
Benchmarking CPU implementation...
Benchmarking GPU implementation...
CPU average time: 1044361.482374 microseconds
GPU average time: 433.850847 microseconds
Speedup: 2407.190142x
```