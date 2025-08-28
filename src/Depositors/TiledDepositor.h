#pragma once
#include "DepositorBase.h"

class TiledDepositor : public DepositorBase {
public:
    void deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter) override;
};

__global__ void deposit_density_2d_tiled(
    float *x, float *y, float *N,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
);

__global__ void deposit_velocity_2d_tiled(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *N, float *Ux, float *Uy,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
);

__global__ void deposit_temperature_2d_tiled(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *N, const float *Ux, const float *Uy,
    float *T,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
);

__global__ void deposit_density_2d_VR_tiled(
    const float *x, const float *y,
    const float *w,
    const float *N,
    float *NVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
);

__global__ void deposit_velocity_2d_VR_tiled(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *w,
    const float *NVR,
    float *UxVR, float *UyVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
);

__global__ void deposit_temperature_2d_VR_tiled(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *w,
    const float *N,
    const float *NVR,
    const float *UxVR, const float *UyVR,
    float *TVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
);