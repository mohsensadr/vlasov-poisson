#include "constants.hpp"
#include <iostream>
#include <fstream>

__global__ void deposit_density_2d(float *x, float *y, float *N, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&N[idx], 1.0f);
    }
}

__global__ void deposit_velocity_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&Ux[idx], vx[i]/N[idx]);
        atomicAdd(&Uy[idx], vy[i]/N[idx]);
    }
}

__global__ void deposit_temperature_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, float *T, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        float energy = (vx[i]-Ux[idx])*(vx[i]-Ux[idx]);
        energy += (vy[i]-Uy[idx])*(vy[i]-Uy[idx]);
        atomicAdd(&T[idx], energy/N[idx]/(2.0f*kb/m));
    }
}

__global__ void deposit_density_2d_VR(float *x, float *y, float *w, float *N, float *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        float Navg = (1.0f*n_particles) / (1.0f*N_GRID_X*N_GRID_Y);
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&NVR[idx], Navg/N[idx] + 1.0f - w[i] );
    }
}

__global__ void deposit_velocity_2d_VR(float *x, float *y, float *vx, float*vy, float *w,
            float *UxVR, float *UyVR, float *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&UxVR[idx], vx[i] * ( 1.0f - w[i] ) / NVR[idx] );
        atomicAdd(&UyVR[idx], vy[i] * ( 1.0f - w[i] ) / NVR[idx] );
    }
}

__global__ void deposit_temperature_2d_VR(float *x, float *y, float *vx, float *vy, float *w, float *N, float *NVR, float *UxVR, float *UyVR, float *TVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        float Navg = (1.0f*n_particles) / (1.0f*N_GRID_X*N_GRID_Y);
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        float energy = (vx[i]-UxVR[idx])*(vx[i]-UxVR[idx]);
        energy += (vy[i]-UyVR[idx])*(vy[i]-UyVR[idx]);
        float ans = Navg/(kb/m)/NVR[idx]/N[idx] + ( energy*(1.0f-w[i])/2.0f ) / (kb/m) / NVR[idx];
        atomicAdd(&TVR[idx], ans);
    }
}

void compute_moments(float *d_x, float *d_y, float *d_vx, float *d_vy, 
    float *d_N, float *d_Ux, float *d_Uy, float *d_T,
    float *d_w, float *d_NVR, float *d_UxVR, float *d_UyVR, float *d_TVR,
    int n_particles, int N_GRID_X, int N_GRID_Y, float Lx, float Ly,
    int blocksPerGrid, int threadsPerBlock){
    
    int grid_size = N_GRID_X*N_GRID_Y;
    
    cudaMemset(d_N, 0, sizeof(float) * grid_size);
    cudaMemset(d_Ux, 0, sizeof(float) * grid_size);
    cudaMemset(d_Uy, 0, sizeof(float) * grid_size);
    cudaMemset(d_T, 0, sizeof(float) * grid_size);

    cudaMemset(d_NVR, 0, sizeof(float) * grid_size);
    cudaMemset(d_UxVR, 0, sizeof(float) * grid_size);
    cudaMemset(d_UyVR, 0, sizeof(float) * grid_size);
    cudaMemset(d_TVR, 0, sizeof(float) * grid_size);

    cudaMemcpyToSymbol(kb, &kb_host, sizeof(float));
    cudaMemcpyToSymbol(m, &m_host, sizeof(float));

    // compute number of particles in each cell (MC)      
    deposit_density_2d<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_N, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute bulk velocity (MC)
    deposit_velocity_2d<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_N, d_vx, d_vy, d_Ux, d_Uy, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute bulk temperature (MC)
    deposit_temperature_2d<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_N, d_vx, d_vy, d_Ux, d_Uy, d_T, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute density (VR)
    deposit_density_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_w, d_N, d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute velocity (VR)
    deposit_velocity_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_vx, d_vy, d_w, d_UxVR, d_UyVR, d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute temperature (VR)
    deposit_temperature_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_vx, d_vy, d_w, d_N, d_NVR, d_UxVR, d_UyVR, d_TVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();
}

