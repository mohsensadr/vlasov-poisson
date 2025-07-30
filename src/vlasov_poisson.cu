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
#include "particle_container.cuh"
#include "field_container.cuh"

/*
struct PDF_position {
    float A;
    float kx;
    float Lx;
    float Ly;

    __device__ float normalizer() const {
        return (A * sinf(Lx * kx) + Lx * kx) / kx;
    }

    __device__ float pmax() const {
        return (1.0f + A) / normalizer() * (1.0f / Ly);
    }

    __device__ float operator()(float x, float y) const {
        return (1.0f + A * cosf(kx * x)) / normalizer() * (1.0f / Ly);
    }

};
*/

struct PDF_position {
    float var;
    float Lx;
    float Ly;

    __device__ float normalizer() const {
        return 2.0*3.14*var;
    }

    __device__ float pmax() const {
        return 1.0;
    }

    __device__ float operator()(float x, float y) const {
        return expf(-(x-Lx/2.0f)*(x-Lx/2.0f)/2.0/var -(y-Ly/2.0f)*(y-Ly/2.0f)/2.0/var);
    }

};


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

void run() {
    cudaMemcpyToSymbol(kb, &kb_host, sizeof(float));
    cudaMemcpyToSymbol(m, &m_host, sizeof(float));
    
    dx = Lx/N_GRID_X;
    dy = Ly/N_GRID_Y;
    grid_size = N_GRID_X*N_GRID_Y;

    ParticleContainer pc(N_PARTICLES);
    FieldContainer fc(N_GRID_X, N_GRID_Y);

    //PDF_position pdf_position{1.0f, 0.5f, Lx, Ly};
    PDF_position pdf_position{Lx*Ly*10.0f, Lx, Ly};

    // initialize particle velocity and position
    initialize_particles<<<blocksPerGrid, threadsPerBlock>>>(
        pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, Lx, Ly, N_PARTICLES, pdf_position
    );
    cudaDeviceSynchronize();

    // compute moments, needed to find emperical density field
    compute_moments(pc, fc);
    cudaDeviceSynchronize();

    // set particle weights given estimted and exact fields
    initialize_weights<<<blocksPerGrid, threadsPerBlock>>>(
        pc.d_x, pc.d_y, fc.d_N, pc.d_w, N_PARTICLES, N_GRID_X, N_GRID_Y, Lx, Ly, pdf_position
    );
    cudaDeviceSynchronize();

    // recompute moments given weights, mainly for VR estimate
    compute_moments(pc, fc);
    cudaDeviceSynchronize();

    // write out initial fields
    post_proc(fc, 0);
    cudaDeviceSynchronize();

    for (int step = 1; step < NSteps+1; ++step) {

        // compute Electric field
        solve_poisson_jacobi(fc);
        cudaDeviceSynchronize();

        // map weights from global to local eq.
        map_weights_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR, N_PARTICLES, N_GRID_X, N_GRID_Y,
            Lx, Ly, true);
        cudaDeviceSynchronize();

        // push particles in the velocity space
        update_velocity_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, fc.d_Ex, fc.d_Ey, N_PARTICLES, N_GRID_X, N_GRID_Y,
            Lx, Ly, DT, Q_OVER_M);
        cudaDeviceSynchronize();

        // map weights from local to global eq.
        map_weights_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR, N_PARTICLES, N_GRID_X, N_GRID_Y,
            Lx, Ly, false);
        cudaDeviceSynchronize();

        // push particles in the position space
        update_position_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, N_PARTICLES, Lx, Ly, DT);
        cudaDeviceSynchronize();

        // update moments
        compute_moments(pc, fc);
        cudaDeviceSynchronize();

        // print output
        if (step % 10 == 0) {
            post_proc(fc, step);
            cudaDeviceSynchronize();
        }
    }
    std::cout << "Done.\n";
}

