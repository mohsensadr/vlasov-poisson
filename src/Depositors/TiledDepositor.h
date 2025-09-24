#pragma once
#include "DepositorBase.h"

class TiledDepositor : public DepositorBase {
public:
    void deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter) override;
};

__global__ void deposit_density_2d_tiled(
    float_type *x, float_type *y, float_type *N,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
);

__global__ void deposit_velocity_2d_tiled(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *N, float_type *Ux, float_type *Uy,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
);

__global__ void deposit_temperature_2d_tiled(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *N, const float_type *Ux, const float_type *Uy,
    float_type *T,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
);

__global__ void deposit_density_2d_VR_tiled(
    const float_type *x, const float_type *y,
    const float_type *w,
    const float_type *N,
    float_type *NVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
);

__global__ void deposit_velocity_2d_VR_tiled(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *w,
    const float_type *NVR,
    float_type *UxVR, float_type *UyVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
);

__global__ void deposit_temperature_2d_VR_tiled(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *w,
    const float_type *N,
    const float_type *NVR,
    const float_type *UxVR, const float_type *UyVR,
    float_type *TVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
);