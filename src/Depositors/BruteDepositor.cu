#include "BruteDepositor.h"

__global__ void deposit_density_2d(float *x, float *y, float *N, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  );

__global__ void deposit_velocity_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  );

__global__ void deposit_temperature_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, float *T, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  );

__global__ void deposit_density_2d_VR(float *x, float *y, float *w, float *N, float *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  );

__global__ void deposit_velocity_2d_VR(float *x, float *y, float *vx, float*vy, float *w,
            float *UxVR, float *UyVR, float *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  );

__global__ void deposit_temperature_2d_VR(float *x, float *y, float *vx, float *vy, float *w, float *N, float *NVR, float *UxVR, float *UyVR, float *TVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  );

void BruteDepositor::deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& /*sorter*/) {
    int n_particles = N_PARTICLES;
    dim3 threads(256);
    dim3 blocks((n_particles + 255) / 256);

    launch(deposit_density_2d, blocks, threads, pc.d_x, pc.d_y, fc.d_N,
           n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    launch(deposit_velocity_2d, blocks, threads, pc.d_x, pc.d_y, fc.d_N,
           pc.d_vx, pc.d_vy, fc.d_Ux, fc.d_Uy, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    launch(deposit_temperature_2d, blocks, threads, pc.d_x, pc.d_y, fc.d_N,
           pc.d_vx, pc.d_vy, fc.d_Ux, fc.d_Uy, fc.d_T, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    launch(deposit_density_2d_VR, blocks, threads, pc.d_x, pc.d_y, pc.d_w, fc.d_N, fc.d_NVR,
           n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    launch(deposit_velocity_2d_VR, blocks, threads, pc.d_x, pc.d_y, pc.d_vx, pc.d_vy,
           pc.d_w, fc.d_UxVR, fc.d_UyVR, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    launch(deposit_temperature_2d_VR, blocks, threads, pc.d_x, pc.d_y, pc.d_vx, pc.d_vy,
           pc.d_w, fc.d_N, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR,
           n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
}

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