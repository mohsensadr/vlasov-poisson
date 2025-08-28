#pragma once
#include "constants.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

class ParticleContainer {
public:
    float *d_x = nullptr;
    float *d_y = nullptr;
    float *d_vx = nullptr;
    float *d_vy = nullptr;
    float *d_w = nullptr;
    float *d_wold = nullptr;
    int n_particles;

    ParticleContainer(int n_particles_);
    ~ParticleContainer();

    void update_velocity(float *Ex, float *Ey,
                         int N_GRID_X, int N_GRID_Y,
                         float Lx, float Ly,
                         float DT, float Q_OVER_M);

    void update_position(float Lx, float Ly, float DT);

    void map_weights(float *NVR, float *UxVR, float *UyVR, float *TVR,
                     int N_GRID_X, int N_GRID_Y, float Lx, float Ly, bool global_to_local);
};

// Kernel declarations only
__global__ void update_velocity_2d(float *x, float *y, float *vx, float *vy,
                                   float *Ex, float *Ey, int n_particles,
                                   int N_GRID_X, int N_GRID_Y,
                                   float Lx, float Ly,
                                   float DT, float Q_OVER_M);

__global__ void update_position_2d(float *x, float *y, float *vx, float *vy,
                                   int n_particles, float Lx, float Ly, float DT);

__global__ void map_weights_2d(float *x, float *y, float *vx, float *vy, float *w,
                               float *NVR, float *UxVR, float *UyVR, float *TVR,
                               int n_particles, int N_GRID_X, int N_GRID_Y,
                               float Lx, float Ly, bool global_to_local);

__device__ int periodic_index(int i, int N);
