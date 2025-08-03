#include "constants.hpp"
#include <iostream>
#include <fstream>
#include "particle_container.cuh"
#include "field_container.cuh"

#define TILE_X 32
#define TILE_Y 32

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

__global__ void deposit_density_2d_tiled(
    float *x, float *y, float *N,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
) {
    // Each block handles a TILE_X x TILE_Y tile of the grid
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    float dx = N_GRID_X / Lx;
    float dy = N_GRID_Y / Ly;

    // Shared memory tile
    __shared__ float tile_counts[TILE_Y][TILE_X];

    // Initialize shared memory
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_counts[local_y][local_x] = 0.0f;
    }
    __syncthreads();

    // Flatten thread index
    int tid = blockIdx.z * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;

    // Stride over particles
    for (int i = tid; i < n_particles; i += gridDim.z * blockDim.x * blockDim.y) {
        float px = x[i];
        float py = y[i];

        int gx = int(px * dx);
        int gy = int(py * dy);

        // Skip particles outside this tile
        if (gx >= tile_origin_x && gx < tile_origin_x + TILE_X &&
            gy >= tile_origin_y && gy < tile_origin_y + TILE_Y) {
            int lx = gx - tile_origin_x;
            int ly = gy - tile_origin_y;
            atomicAdd(&tile_counts[ly][lx], 1.0f);
        }
    }

    __syncthreads();

    // Write tile results back to global memory
    int gx = tile_origin_x + threadIdx.x;
    int gy = tile_origin_y + threadIdx.y;
    if (threadIdx.x < TILE_X && threadIdx.y < TILE_Y &&
        gx < N_GRID_X && gy < N_GRID_Y) {
        int idx = gx + gy * N_GRID_X;
        atomicAdd(&N[idx], tile_counts[threadIdx.y][threadIdx.x]);
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

__global__ void deposit_velocity_2d_tiled(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *N, float *Ux, float *Uy,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
) {
    // Compute origin of the tile in grid space
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    // Grid scale
    float dx = N_GRID_X / Lx;
    float dy = N_GRID_Y / Ly;

    // Allocate shared memory tile
    __shared__ float tile_Ux[TILE_Y][TILE_X];
    __shared__ float tile_Uy[TILE_Y][TILE_X];

    // Zero shared memory (by all threads in block)
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_Ux[local_y][local_x] = 0.0f;
        tile_Uy[local_y][local_x] = 0.0f;
    }
    __syncthreads();

    // Flatten thread ID across block
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int i = thread_id; i < n_particles; i += threads_per_block) {
        float px = x[i];
        float py = y[i];

        int gx = int(px * dx);
        int gy = int(py * dy);

        // Clamp to grid bounds to prevent overflow
        if (gx >= 0 && gx < N_GRID_X &&
            gy >= 0 && gy < N_GRID_Y) {

                          // Check if particle belongs in this tile
            if (gx >= tile_origin_x && gx < tile_origin_x + TILE_X &&
                gy >= tile_origin_y && gy < tile_origin_y + TILE_Y) {

                int lx = gx - tile_origin_x;
                int ly = gy - tile_origin_y;

                int idx = gx + gy * N_GRID_X;

                float nval = N[idx];  // read global N[idx]
                if (nval > 0.0f) {
                    float vx_scaled = vx[i] / nval;
                    float vy_scaled = vy[i] / nval;

                    atomicAdd(&tile_Ux[ly][lx], vx_scaled);
                    atomicAdd(&tile_Uy[ly][lx], vy_scaled);
                }
            }
        }
    }

    __syncthreads();

        // Write accumulated shared memory back to global memory
    int gx = tile_origin_x + threadIdx.x;
    int gy = tile_origin_y + threadIdx.y;

    if (threadIdx.x < TILE_X && threadIdx.y < TILE_Y &&
        gx < N_GRID_X && gy < N_GRID_Y) {

        int idx = gx + gy * N_GRID_X;
        atomicAdd(&Ux[idx], tile_Ux[threadIdx.y][threadIdx.x]);
        atomicAdd(&Uy[idx], tile_Uy[threadIdx.y][threadIdx.x]);
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

__global__ void deposit_temperature_2d_tiled(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *N, const float *Ux, const float *Uy,
    float *T,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
) {
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    float dx = float(N_GRID_X) / Lx;
    float dy = float(N_GRID_Y) / Ly;

    __shared__ float tile_T[TILE_Y][TILE_X];

    // Initialize shared memory
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_T[local_y][local_x] = 0.0f;
    }
    __syncthreads();

    // Flatten thread ID across block and z dimension
    int tid = blockIdx.z * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = gridDim.z * blockDim.x * blockDim.y;

    for (int i = tid; i < n_particles; i += total_threads) {
        float px = x[i];
        float py = y[i];

        int gx = int(px * dx);
        int gy = int(py * dy);

        // Clamp to grid bounds
        gx = min(max(gx, 0), N_GRID_X - 1);
        gy = min(max(gy, 0), N_GRID_Y - 1);

        // Check if this grid cell belongs to current tile
        if (gx >= tile_origin_x && gx < tile_origin_x + TILE_X &&
            gy >= tile_origin_y && gy < tile_origin_y + TILE_Y) {

            int lx = gx - tile_origin_x;
            int ly = gy - tile_origin_y;
            int idx = gx + gy * N_GRID_X;

            float nval = N[idx];
            if (nval > 0.0f) {
                float ux = Ux[idx];
                float uy = Uy[idx];
                float dvx = vx[i] - ux;
                float dvy = vy[i] - uy;
                float energy = (dvx * dvx + dvy * dvy);
                float temp_contrib = energy / nval / (2.0f * kb / m);

                atomicAdd(&tile_T[ly][lx], temp_contrib);
            }
        }
    }

    __syncthreads();

    // Write tile results back to global memory
    int gx = tile_origin_x + threadIdx.x;
    int gy = tile_origin_y + threadIdx.y;

    if (threadIdx.x < TILE_X && threadIdx.y < TILE_Y &&
        gx < N_GRID_X && gy < N_GRID_Y) {
        int idx = gx + gy * N_GRID_X;
        atomicAdd(&T[idx], tile_T[threadIdx.y][threadIdx.x]);
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

__global__ void deposit_density_2d_VR_tiled(
    const float *x, const float *y,
    const float *w,
    const float *N,
    float *NVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
) {
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    float dx = float(N_GRID_X) / Lx;
    float dy = float(N_GRID_Y) / Ly;

    float Navg = (1.0f * n_particles) / (N_GRID_X * N_GRID_Y);

    __shared__ float tile_NVR[TILE_Y][TILE_X];

    // Initialize shared memory
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_NVR[local_y][local_x] = 0.0f;
    }
    __syncthreads();

    // Flatten thread ID
    int tid = blockIdx.z * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = gridDim.z * blockDim.x * blockDim.y;

    for (int i = tid; i < n_particles; i += total_threads) {
        float px = x[i];
        float py = y[i];

        int gx = int(px * dx);
        int gy = int(py * dy);

        // Clamp to grid
        gx = min(max(gx, 0), N_GRID_X - 1);
        gy = min(max(gy, 0), N_GRID_Y - 1);

        if (gx >= tile_origin_x && gx < tile_origin_x + TILE_X &&
            gy >= tile_origin_y && gy < tile_origin_y + TILE_Y) {

            int lx = gx - tile_origin_x;
            int ly = gy - tile_origin_y;
            int idx = gx + gy * N_GRID_X;

            float nval = N[idx];
            if (nval > 0.0f) {
                float delta = Navg / nval + 1.0f - w[i];
                atomicAdd(&tile_NVR[ly][lx], delta);
            }
        }
    }

    __syncthreads();

    // Write back to global memory
    int gx = tile_origin_x + threadIdx.x;
    int gy = tile_origin_y + threadIdx.y;

    if (threadIdx.x < TILE_X && threadIdx.y < TILE_Y &&
        gx < N_GRID_X && gy < N_GRID_Y) {
        int idx = gx + gy * N_GRID_X;
        atomicAdd(&NVR[idx], tile_NVR[threadIdx.y][threadIdx.x]);
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

void compute_moments(ParticleContainer& pc, FieldContainer& fc){
    int n_particles = N_PARTICLES;

    cudaMemcpyToSymbol(kb, &kb_host, sizeof(float));
    cudaMemcpyToSymbol(m, &m_host, sizeof(float));

    fc.setZero();

    dim3 threadsPerBlock2d(TILE_X, TILE_Y);
    dim3 blocksPerGrid2d(
        (N_GRID_X + TILE_X - 1) / TILE_X,
        (N_GRID_Y + TILE_Y - 1) / TILE_Y,
        1  // You can parallelize over z if needed
    );
    deposit_density_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, fc.d_N, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute number of particles in each cell (MC)      
    //deposit_density_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, fc.d_N, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    //cudaDeviceSynchronize();

    // compute bulk velocity (MC)
    deposit_velocity_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, fc.d_N, fc.d_Ux, fc.d_Uy, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    //deposit_velocity_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, fc.d_N, pc.d_vx, pc.d_vy, fc.d_Ux, fc.d_Uy, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute bulk temperature (MC)
    //deposit_temperature_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, fc.d_N, pc.d_vx, pc.d_vy, fc.d_Ux, fc.d_Uy, fc.d_T, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    deposit_temperature_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, fc.d_N, fc.d_Ux, fc.d_Uy, fc.d_T, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute density (VR)
    //deposit_density_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_w, fc.d_N, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    deposit_density_2d_VR_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_w, fc.d_N, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute velocity (VR)
    deposit_velocity_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_UxVR, fc.d_UyVR, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute temperature (VR)
    deposit_temperature_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_N, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();
    
}

