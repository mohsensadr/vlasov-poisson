#pragma once
#include "Constants/constants.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

class ParticleContainer {
public:
    float_type *d_x = nullptr;
    float_type *d_y = nullptr;
    float_type *d_vx = nullptr;
    float_type *d_vy = nullptr;
    float_type *d_vx_old = nullptr;
    float_type *d_vy_old = nullptr;
    float_type *d_w = nullptr;
    float_type *d_wold = nullptr;
    int n_particles;

    ParticleContainer(int n_particles_);
    ~ParticleContainer();

    void update_velocity(float_type *Ex, float_type *Ey,
                         int N_GRID_X, int N_GRID_Y,
                         float_type Lx, float_type Ly,
                         float_type DT, float_type Q_OVER_M);

    void update_position(float_type Lx, float_type Ly, float_type DT);

    void save_old_velocity();

    void map_weights(float_type *NVR, float_type *UxVR, float_type *UyVR, float_type *TVR,
                     int N_GRID_X, int N_GRID_Y, float_type Lx, float_type Ly, bool global_to_local);
};

// Kernel declarations only
__global__ void update_velocity_2d(float_type *x, float_type *y, float_type *vx, float_type *vy,
                                   float_type *Ex, float_type *Ey, int n_particles,
                                   int N_GRID_X, int N_GRID_Y,
                                   float_type Lx, float_type Ly,
                                   float_type DT, float_type Q_OVER_M);

__global__ void update_position_2d(float_type *x, float_type *y, float_type *vx, float_type *vy,
                                   int n_particles, float_type Lx, float_type Ly, float_type DT);

__global__ void map_weights_2d(float_type *x, float_type *y, float_type *vx, float_type *vy, float_type *w,
                               float_type *NVR, float_type *UxVR, float_type *UyVR, float_type *TVR,
                               int n_particles, int N_GRID_X, int N_GRID_Y,
                               float_type Lx, float_type Ly, bool global_to_local);

__global__ void save_old_velocity_2d(float_type *vx, float_type *vy,
                              float_type *vx_old, float_type *vy_old, int n_particles);

__device__ int periodic_index(int i, int N);
