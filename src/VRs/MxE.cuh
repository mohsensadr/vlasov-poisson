#pragma once
#include <cuda_runtime.h>
#include "Constants/constants.hpp"

template <typename T>
struct Tolerance;

template <>
struct Tolerance<float> {
    static __device__ __host__ inline float_type value() { return 1e-3f; }
};

template <>
struct Tolerance<double> {
    static __device__ __host__ inline double value() { return 1e-9; }
};

template<int Nm>
__device__ void Gauss_Jordan(float_type H[Nm][Nm], float_type g[Nm], float_type x[Nm]);

template<int Nm>
__global__ void update_weights(
    const float_type* __restrict__ vx,
    const float_type* __restrict__ vy,
    const float_type* __restrict__ vx_old,
    const float_type* __restrict__ vy_old,
    const int* __restrict__ d_cell_offsets,
    float_type* __restrict__ w,
    float_type* __restrict__ wold,
    float_type* __restrict__ NVR,
    float_type* __restrict__ UxVR,
    float_type* __restrict__ UyVR,
    float_type* __restrict__ ExVR,
    float_type* __restrict__ EyVR,
    int num_cells,
    int n_particles,
    float_type dt
);

void update_weights_dispatch(
    const float_type* vx,
    const float_type* vy,
    const float_type* vx_old,
    const float_type* vy_old,
    const int* d_cell_offsets,
    float_type* w,
    float_type* wold,
    float_type* NVR,
    float_type* UxVR,
    float_type* UyVR,
    float_type* ExVR,
    float_type* EyVR,
    int num_cells,
    int Nm
);