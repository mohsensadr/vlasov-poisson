#ifndef PARTICLE_INITIALIZER_H
#define PARTICLE_INITIALIZER_H

#include <cuda_runtime.h>

/**
 * @brief CUDA kernel to initialize particles on the GPU.
 *        Each particle is assigned:
 *        - position (x, y) cos(x) distriution
 *        - velocity (vx, vy) from Maxwellian using Box-Muller transform
 *        - small Landau density perturbation added to velocity
 *
 * @param x   [out] Particle x positions (device pointer)
 * @param y   [out] Particle y positions (device pointer)
 * @param vx  [out] Particle x velocities (device pointer)
 * @param vy  [out] Particle y velocities (device pointer)
 * @param Lx  Domain width
 * @param Ly  Domain height
 * @param N   Number of particles
 */
__global__ void initialize_particles(float *x, float *y,
                                     float *vx, float *vy,
                                     float Lx, float Ly,
                                     int N, float A=1.0, float kx=0.5);

__global__ void initialize_weights(float *x, float *y, float *N,float *w,
        int Ntotal, int N_GRID_X, int N_GRID_Y, float Lx, float Ly, float A=1.0f, float kx=0.6f);

#endif  // PARTICLE_INITIALIZER_H
