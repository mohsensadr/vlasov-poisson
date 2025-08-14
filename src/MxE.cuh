#pragma once
#include <cuda_runtime.h>

template<int Nm>
__device__ void Gauss_Jordan(float H[Nm][Nm], float g[Nm], float x[Nm]);

template<int Nm>
__device__ float mom(float u1, float u2, float U_1, float U_2, int n);

template<int Nm>
__global__ void update_weights(
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const int* __restrict__ d_cell_offsets,
    float* __restrict__ w,
    float* __restrict__ wold,
    float* __restrict__ UxVR,
    float* __restrict__ UyVR,
    int num_cells
);

