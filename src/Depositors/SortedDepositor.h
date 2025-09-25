#pragma once
#include "DepositorBase.h"

class SortedDepositor : public DepositorBase {
public:
    void deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter) override;
};

__global__ void deposit_density_2d_sorted(
    const int* __restrict__ d_cell_counts,
    float_type* __restrict__ d_N,
    int num_cells
);

__global__ void deposit_velocity_2d_sorted(
    const float_type* __restrict__ vx,
    const float_type* __restrict__ vy,
    const int* __restrict__ d_cell_offsets,
    float_type* __restrict__ Ux,
    float_type* __restrict__ Uy,
    int num_cells
);

__global__ void deposit_temperature_2d_sorted(
    const float_type* __restrict__ vx,
    const float_type* __restrict__ vy,
    const int*   __restrict__ d_cell_offsets, // size: num_cells + 1
    const float_type* __restrict__ Ux,
    const float_type* __restrict__ Uy,
    float_type* T,
    int num_cells
);

__global__ void deposit_density_2d_VR_sorted(
    const float_type* __restrict__ w,          // particle weights
    const int*   __restrict__ d_cell_offsets, // per-cell start indices (size num_cells+1)
    float_type* NVR,                           // output: variance-reduced density
    int num_cells,
    int n_particles
);

__global__ void deposit_velocity_2d_VR_sorted(
    const float_type* __restrict__ vx,
    const float_type* __restrict__ vy,
    const float_type* __restrict__ w,
    const int*   __restrict__ d_cell_offsets, // start indices of particles per cell
    const float_type* __restrict__ NVR,            // number of particles per cell / density
    float_type* UxVR,                              // output: x-velocity per cell
    float_type* UyVR,                              // output: y-velocity per cell
    int num_cells
);

__global__ void deposit_temperature_2d_VR_sorted(
    const float_type* __restrict__ vx,
    const float_type* __restrict__ vy,
    const float_type* __restrict__ w,
    const float_type* __restrict__ UxVR,
    const float_type* __restrict__ UyVR,
    const int*   __restrict__ d_cell_offsets, // start indices of particles per cell
    const float_type* __restrict__ NVR,            // VR density
    float_type* TVR,                               // output: VR temperature per cell
    int num_cells
);