#pragma once
#include "DepositorBase.h"

class SortedDepositor : public DepositorBase {
public:
    void deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter) override;
};

__global__ void deposit_density_2d_sorted(
    const int* __restrict__ d_cell_counts,
    float* __restrict__ d_N,
    int num_cells
);

__global__ void deposit_velocity_2d_sorted(
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const int* __restrict__ d_cell_offsets,
    float* __restrict__ Ux,
    float* __restrict__ Uy,
    int num_cells
);

__global__ void deposit_temperature_2d_sorted(
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const int*   __restrict__ d_cell_offsets, // size: num_cells + 1
    const float* __restrict__ Ux,
    const float* __restrict__ Uy,
    float* T,
    int num_cells
);

__global__ void deposit_density_2d_VR_sorted(
    const float* __restrict__ w,          // particle weights
    const int*   __restrict__ d_cell_offsets, // per-cell start indices (size num_cells+1)
    float* NVR,                           // output: variance-reduced density
    int num_cells,
    int n_particles
);

__global__ void deposit_velocity_2d_VR_sorted(
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ w,
    const int*   __restrict__ d_cell_offsets, // start indices of particles per cell
    const float* __restrict__ NVR,            // number of particles per cell / density
    float* UxVR,                              // output: x-velocity per cell
    float* UyVR,                              // output: y-velocity per cell
    int num_cells
);

__global__ void deposit_temperature_2d_VR_sorted(
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ w,
    const float* __restrict__ UxVR,
    const float* __restrict__ UyVR,
    const int*   __restrict__ d_cell_offsets, // start indices of particles per cell
    const float* __restrict__ NVR,            // VR density
    float* TVR,                               // output: VR temperature per cell
    int num_cells
);