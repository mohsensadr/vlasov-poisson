#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#include "constants.hpp"
#include "solver.cuh"
#include "initialization.cuh"
#include "IO.h"
#include "moments.cuh"

static __device__ int periodic_index(int i, int N) {
    return (i + N) % N;
}

__global__ void update_velocity_2d(float *x, float *y, float *vx, float *vy,
                                  float *Ex, float *Ey, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly,
            float DT,
            float Q_OVER_M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    float xi = x[i] / Lx * N_GRID_X;
    float yi = y[i] / Ly * N_GRID_Y;

    int ix = floorf(xi);
    int iy = floorf(yi);
    float dx = xi - ix;
    float dy = yi - iy;

    int ix0 = periodic_index(ix, N_GRID_X);
    int ix1 = periodic_index(ix + 1, N_GRID_X);
    int iy0 = periodic_index(iy, N_GRID_Y);
    int iy1 = periodic_index(iy + 1, N_GRID_Y);

    float w00 = (1 - dx) * (1 - dy);
    float w01 = (1 - dx) * dy;
    float w10 = dx * (1 - dy);
    float w11 = dx * dy;

    int i00 = ix0 + iy0 * N_GRID_X;
    int i01 = ix0 + iy1 * N_GRID_X;
    int i10 = ix1 + iy0 * N_GRID_X;
    int i11 = ix1 + iy1 * N_GRID_X;

    float Exi = w00 * Ex[i00] + w01 * Ex[i01] + w10 * Ex[i10] + w11 * Ex[i11];
    float Eyi = w00 * Ey[i00] + w01 * Ey[i01] + w10 * Ey[i10] + w11 * Ey[i11];

    vx[i] += - Q_OVER_M * Exi * DT;
    vy[i] += - Q_OVER_M * Eyi * DT;
}

__global__ void update_position_2d(float *x, float *y, float *vx, float *vy,
                                  int n_particles, float Lx, float Ly, float DT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    x[i] += vx[i] * DT;
    y[i] += vy[i] * DT;

    // periodic boundaries
    if (x[i] < 0) x[i] += Lx;
    if (x[i] >= Lx) x[i] -= Lx;
    if (y[i] < 0) y[i] += Ly;
    if (y[i] >= Ly) y[i] -= Ly;
}

__global__ void map_weights_2d(float *x, float *y, float *vx, float *vy, float *w,
    float *NVR, float *UxVR, float *UyVR, float *TVR, int n_particles,
    int N_GRID_X, int N_GRID_Y, float Lx, float Ly, bool global_to_local
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        float Navg = (1.0f*n_particles) / (1.0f*N_GRID_X*N_GRID_Y);
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        float energy = (vx[i]-UxVR[idx])*(vx[i]-UxVR[idx]);
        energy += (vy[i]-UyVR[idx])*(vy[i]-UyVR[idx]);
        float energy0 = vx[i]*vx[i] + vy[i]*vy[i];
        float kbT_m = kb*TVR[idx]/m;
        float kbT_m0 = kb*1.0f/m;
        if(global_to_local){
          w[i] = w[i] * (NVR[idx]/Navg) * (kbT_m0/kbT_m) * expf(-energy/kbT_m/2.0f+energy0/kbT_m0/2.0f);
        }
        else{
          w[i] = w[i] * (Navg/NVR[idx]) * (kbT_m/kbT_m0) * expf(energy/kbT_m/2.0f-energy0/kbT_m0/2.0f);
        }
    }
}

void run(int N_GRID_X, int N_GRID_Y,
            int N_PARTICLES,
            float DT,
            int NSteps,
            float Lx,
            float Ly,
            int threadsPerBlock
            ) {
    cudaMemcpyToSymbol(kb, &kb_host, sizeof(float));
    cudaMemcpyToSymbol(m, &m_host, sizeof(float));
    
    float dx = Lx/N_GRID_X;
    float dy = Ly/N_GRID_Y;
    int grid_size = N_GRID_X*N_GRID_Y;
    float *d_x, *d_y, *d_vx, *d_vy;
    float *d_N, *d_Ux, *d_Uy, *d_T, *d_Ex, *d_Ey;
    float *d_w, *d_NVR, *d_UxVR, *d_UyVR, *d_TVR;

    cudaMalloc(&d_x, sizeof(float) * N_PARTICLES);
    cudaMalloc(&d_y, sizeof(float) * N_PARTICLES);
    cudaMalloc(&d_vx, sizeof(float) * N_PARTICLES);
    cudaMalloc(&d_vy, sizeof(float) * N_PARTICLES);
    cudaMalloc(&d_w, sizeof(float) * N_PARTICLES);

    cudaMalloc(&d_N, sizeof(float) * N_GRID_X * N_GRID_Y);
    cudaMalloc(&d_Ux, sizeof(float) * N_GRID_X * N_GRID_Y);
    cudaMalloc(&d_Uy, sizeof(float) * N_GRID_X * N_GRID_Y);
    cudaMalloc(&d_T, sizeof(float) * N_GRID_X * N_GRID_Y);
    cudaMalloc(&d_Ex, sizeof(float) * N_GRID_X * N_GRID_Y);
    cudaMalloc(&d_Ey, sizeof(float) * N_GRID_X * N_GRID_Y);

    cudaMalloc(&d_NVR, sizeof(float) * N_GRID_X * N_GRID_Y);
    cudaMalloc(&d_UxVR, sizeof(float) * N_GRID_X * N_GRID_Y);
    cudaMalloc(&d_UyVR, sizeof(float) * N_GRID_X * N_GRID_Y);
    cudaMalloc(&d_TVR, sizeof(float) * N_GRID_X * N_GRID_Y);

    int blocksPerGrid = (N_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;

    // initialize particle velocity and position
    initialize_particles<<<blocksPerGrid, threadsPerBlock>>>(
        d_x, d_y, d_vx, d_vy, Lx, Ly, N_PARTICLES
    );
    cudaDeviceSynchronize();

    // compute moments, needed to find emperical density field
    compute_moments(d_x, d_y, d_vx, d_vy, d_N, d_Ux, d_Uy, d_T, d_w, d_NVR, d_UxVR, d_UyVR, d_TVR,
        N_PARTICLES, N_GRID_X, N_GRID_Y, Lx, Ly, blocksPerGrid, threadsPerBlock);
    cudaDeviceSynchronize();

    // set particle weights given estimted and exact fields
    initialize_weights<<<blocksPerGrid, threadsPerBlock>>>(
        d_x, d_y, d_N, d_w, N_PARTICLES, N_GRID_X, N_GRID_Y, Lx, Ly
    );
    cudaDeviceSynchronize();

    // recompute moments given weights, mainly for VR estimate
    compute_moments(d_x, d_y, d_vx, d_vy, d_N, d_Ux, d_Uy, d_T, d_w, d_NVR, d_UxVR, d_UyVR, d_TVR,
        N_PARTICLES, N_GRID_X, N_GRID_Y, Lx, Ly, blocksPerGrid, threadsPerBlock);
    cudaDeviceSynchronize();

    // write out initial fields
    post_proc(d_N, d_Ux, d_Uy, d_T, d_NVR, d_UxVR, d_UyVR, d_TVR, grid_size, 0);
    cudaDeviceSynchronize();

    for (int step = 1; step < NSteps+1; ++step) {

        // compute Electric field
        solve_poisson_jacobi(d_N, d_Ex, d_Ey, N_GRID_X, N_GRID_Y, dx, dy, threadsPerBlock);
        cudaDeviceSynchronize();

        // map weights from global to local eq.
        map_weights_2d<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_vx, d_vy, d_w, d_NVR, d_UxVR, d_UyVR, d_TVR, N_PARTICLES, N_GRID_X, N_GRID_Y,
            Lx, Ly, true);
        cudaDeviceSynchronize();

        // push particles in the velocity space
        update_velocity_2d<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_vx, d_vy, d_Ex, d_Ey, N_PARTICLES, N_GRID_X, N_GRID_Y,
            Lx, Ly, DT, Q_OVER_M);
        cudaDeviceSynchronize();

        // map weights from local to global eq.
        map_weights_2d<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_vx, d_vy, d_w, d_NVR, d_UxVR, d_UyVR, d_TVR, N_PARTICLES, N_GRID_X, N_GRID_Y,
            Lx, Ly, false);
        cudaDeviceSynchronize();

        // push particles in the position space
        update_position_2d<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_vx, d_vy, N_PARTICLES, Lx, Ly, DT);
        cudaDeviceSynchronize();

        // update moments
        compute_moments(d_x, d_y, d_vx, d_vy, d_N, d_Ux, d_Uy, d_T, d_w, d_NVR, d_UxVR, d_UyVR, d_TVR,
            N_PARTICLES, N_GRID_X, N_GRID_Y, Lx, Ly, blocksPerGrid, threadsPerBlock);
        cudaDeviceSynchronize();

        // print output
        if (step % 10 == 0) {
            post_proc(d_N, d_Ux, d_Uy, d_T, d_NVR, d_UxVR, d_UyVR, d_TVR, grid_size, step);
            cudaDeviceSynchronize();
        }
    }

    // free memory
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_vx); cudaFree(d_vy);
    cudaFree(d_N); cudaFree(d_Ux); cudaFree(d_Uy);cudaFree(d_T);
    cudaFree(d_Ex); cudaFree(d_Ey);
    cudaFree(d_w); cudaFree(d_NVR); cudaFree(d_UxVR); cudaFree(d_UyVR);
    cudaFree(d_TVR);

    std::cout << "Done.\n";
}

