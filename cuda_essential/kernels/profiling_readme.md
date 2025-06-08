# profiling

To install nsys (for CUDA 12.4)
```bash
apt update
apt install nsight-systems-2024.6.2
```

check with `nsys --version`.


# matmul_v1

```bash
Performing warm-up runs...
Benchmarking CPU implementation...
Benchmarking GPU implementation...
CPU average time: 1044361.482374 microseconds
GPU average time: 433.850847 microseconds
Speedup: 2407.190142x
```
`nsys profile -t cuda --stats=true ./matmul_v1` tells 
79.4% time in cudaMalloc



