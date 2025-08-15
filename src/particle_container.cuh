// particle_container.cuh
#pragma once
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

    ParticleContainer(int n_particles_) : n_particles(n_particles_) {
        size_t bytes = n_particles * sizeof(float);
        cudaMalloc(&d_x, bytes);
        cudaMalloc(&d_y, bytes);
        cudaMalloc(&d_vx, bytes);
        cudaMalloc(&d_vy, bytes);
        cudaMalloc(&d_w, bytes);
        cudaMalloc(&d_wold, bytes);
    }

    ~ParticleContainer() {
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_vx);
        cudaFree(d_vy);
        cudaFree(d_w);
        cudaFree(d_wold);
    }
};
