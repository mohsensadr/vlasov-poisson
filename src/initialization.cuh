#ifndef PARTICLE_INITIALIZER_H
#define PARTICLE_INITIALIZER_H

#include <cuda_runtime.h>

// Device function: 2D normalized probability density function
__device__ float pdf(float x, float y, float Lx, float Ly, float A, float kx);

// Kernel: Initializes particle positions and velocities using rejection sampling and normal distribution
__global__ void initialize_particles(float *x, float *y,
                                     float *vx, float *vy,
                                     float Lx, float Ly,
                                     int N, float A=1.0f, float kx=0.6f);

// Kernel: Initializes particle weights to adjust empirical particle density to target PDF
__global__ void initialize_weights(float *x, float *y, float *N, float *w,
                                   int Ntotal, int N_GRID_X, int N_GRID_Y,
                                   float Lx, float Ly,
                                  float A=1.0f, float kx=0.6f);
#endif  // PARTICLE_INITIALIZER_H
