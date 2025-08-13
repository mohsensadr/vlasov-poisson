#include "constants.hpp"
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "moments.cuh"
#include "sorting.cuh"

__global__ void copy_counts_to_density(
    const int* __restrict__ cell_counts,
    float* __restrict__ density,
    int num_cells
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;
    density[i] = static_cast<float>(cell_counts[i]);
}

void compute_density(Sorting& sorter, cudaStream_t stream = 0) {
    int num_cells = sorter.nx * sorter.ny;

    // cell_counts was already computed in sort_particles_and_compute_density()
    // so we just copy it into the density field.
    copy_counts_to_density<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        sorter.d_cell_counts,
        sorter.fc->d_N,
        num_cells
    );

    // optional: sync if you need immediate access to density
    cudaStreamSynchronize(stream);
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

__global__ void deposit_velocity_2d_sorted(
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const int* __restrict__ d_cell_offsets,
    float* __restrict__ Ux,
    float* __restrict__ Uy,
    int num_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    // Get start and end index for this cell
    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1];

    float sum_vx = 0.0f;
    float sum_vy = 0.0f;
    int count = end - start;

    // Sum over particles in this cell
    for (int i = start; i < end; i++) {
        sum_vx += vx[i];
        sum_vy += vy[i];
    }

    // Store average velocity (avoid division by zero)
    if (count > 0) {
        Ux[cell] = sum_vx / count;
        Uy[cell] = sum_vy / count;
    } else {
        Ux[cell] = 0.0f;
        Uy[cell] = 0.0f;
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

// T = ( <(vx-Ux)^2 + (vy-Uy)^2> ) / (2 * kb/m)
__global__ void deposit_temperature_2d_sorted(
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const int*   __restrict__ d_cell_offsets, // size: num_cells + 1
    const float* __restrict__ Ux,
    const float* __restrict__ Uy,
    float* T,
    int num_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1]; // exclusive

    float ux = Ux[cell];
    float uy = Uy[cell];
    float temp_sum = 0.0f;
    int npart = end - start;

    for (int i = start; i < end; ++i) {
        float dvx = vx[i] - ux;
        float dvy = vy[i] - uy;
        temp_sum += dvx * dvx + dvy * dvy;
    }

    T[cell] = (npart > 0) ? temp_sum / (2.0f * kb/m * npart) : 0.0f;
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

__global__ void deposit_density_2d_VR_sorted(
    const float* __restrict__ w,          // particle weights
    const int*   __restrict__ d_cell_offsets, // per-cell start indices (size num_cells+1)
    float* NVR,                           // output: variance-reduced density
    int num_cells,
    int n_particles
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1]; // exclusive
    int npart = end - start;

    float Navg = float(n_particles) / float(num_cells);
    float sum = 0.0f;

    for (int i = start; i < end; ++i) {
        sum += 1.0f - w[i];
    }

    NVR[cell] = (npart > 0) ? Navg + sum : Navg;
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

__global__ void deposit_velocity_2d_VR_sorted(
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ w,
    const int*   __restrict__ d_cell_offsets, // start indices of particles per cell
    const float* __restrict__ NVR,            // number of particles per cell / density
    float* UxVR,                              // output: x-velocity per cell
    float* UyVR,                              // output: y-velocity per cell
    int num_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1]; // exclusive
    int npart = end - start;

    float sum_vx = 0.0f;
    float sum_vy = 0.0f;

    for (int i = start; i < end; ++i) {
        float factor = 1.0f - w[i];
        sum_vx += vx[i] * factor;
        sum_vy += vy[i] * factor;
    }

    // Avoid division by zero
    if (npart > 0) {
        UxVR[cell] = sum_vx / NVR[cell];
        UyVR[cell] = sum_vy / NVR[cell];
    } else {
        UxVR[cell] = 0.0f;
        UyVR[cell] = 0.0f;
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

__global__ void deposit_velocity_2d_VR_tiled(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *w,
    const float *NVR,
    float *UxVR, float *UyVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
) {
    // Tile origin in grid coordinates
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    float dx = float(N_GRID_X) / Lx;
    float dy = float(N_GRID_Y) / Ly;

    // Shared memory tiles
    __shared__ float tile_UxVR[TILE_Y][TILE_X];
    __shared__ float tile_UyVR[TILE_Y][TILE_X];

    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    // Initialize shared memory
    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_UxVR[local_y][local_x] = 0.0f;
        tile_UyVR[local_y][local_x] = 0.0f;
    }
    __syncthreads();

    // Global thread index
    int tid = blockIdx.z * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = gridDim.z * blockDim.x * blockDim.y;

    // Loop over particles
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

            float nvr = NVR[idx];
            if (nvr > 0.0f) {
                float weight_factor = (1.0f - w[i]) / nvr;
                float vx_scaled = vx[i] * weight_factor;
                float vy_scaled = vy[i] * weight_factor;

                atomicAdd(&tile_UxVR[ly][lx], vx_scaled);
                atomicAdd(&tile_UyVR[ly][lx], vy_scaled);
            }
        }
    }

    __syncthreads();

    // Write shared memory back to global
    int gx = tile_origin_x + threadIdx.x;
    int gy = tile_origin_y + threadIdx.y;

    if (gx < N_GRID_X && gy < N_GRID_Y &&
        threadIdx.x < TILE_X && threadIdx.y < TILE_Y) {

        int idx = gx + gy * N_GRID_X;
        atomicAdd(&UxVR[idx], tile_UxVR[threadIdx.y][threadIdx.x]);
        atomicAdd(&UyVR[idx], tile_UyVR[threadIdx.y][threadIdx.x]);
    }
}

__global__ void deposit_temperature_2d_VR_sorted(
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ w,
    const float* __restrict__ UxVR,
    const float* __restrict__ UyVR,
    const int*   __restrict__ d_cell_offsets, // start indices of particles per cell
    const float* __restrict__ NVR,            // VR density
    float* TVR,                               // output: VR temperature per cell
    int num_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1]; // exclusive
    int npart = end - start;

    float temp_sum = 0.0f;

    // Navg for variance reduction
    float Navg = 0.0f;
    if (num_cells > 0) {
        Navg = float(d_cell_offsets[num_cells]) / float(num_cells); // total_particles / num_cells
    }

    float energy;

    for (int i = start; i < end; ++i) {
        float dvx = vx[i] - UxVR[cell];
        float dvy = vy[i] - UyVR[cell];
        energy = (dvx*dvx + dvy*dvy) * 0.5f * (1.0f - w[i]);
    }

    temp_sum = energy / npart / NVR[cell] / (kb/m); // divide by VR density

    // Add eq. term
    if (npart > 0.0f) {
        temp_sum += Navg/(kb/m)/NVR[cell];
    }

    TVR[cell] = temp_sum;
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

__global__ void deposit_temperature_2d_VR_tiled(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *w,
    const float *N,
    const float *NVR,
    const float *UxVR, const float *UyVR,
    float *TVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
) {
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    float dx = float(N_GRID_X) / Lx;
    float dy = float(N_GRID_Y) / Ly;

    // Shared memory tile
    __shared__ float tile_TVR[TILE_Y][TILE_X];

    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_TVR[local_y][local_x] = 0.0f;
    }
    __syncthreads();

    int tid = blockIdx.z * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = gridDim.z * blockDim.x * blockDim.y;

    float Navg = float(n_particles) / (float(N_GRID_X) * float(N_GRID_Y));
    const float coeff = 1.0f / (kb / m); // precompute inverse thermal energy unit

    for (int i = tid; i < n_particles; i += total_threads) {
        float px = x[i];
        float py = y[i];

        int gx = int(px * dx);
        int gy = int(py * dy);

        // Clamp
        gx = min(max(gx, 0), N_GRID_X - 1);
        gy = min(max(gy, 0), N_GRID_Y - 1);

        if (gx >= tile_origin_x && gx < tile_origin_x + TILE_X &&
            gy >= tile_origin_y && gy < tile_origin_y + TILE_Y) {

            int lx = gx - tile_origin_x;
            int ly = gy - tile_origin_y;
            int idx = gx + gy * N_GRID_X;

            float n = N[idx];
            float nvr = NVR[idx];

            if (n > 0.0f && nvr > 0.0f) {
                float ux = UxVR[idx];
                float uy = UyVR[idx];

                float dvx = vx[i] - ux;
                float dvy = vy[i] - uy;

                float energy = 0.5f * (dvx * dvx + dvy * dvy) * (1.0f - w[i]);

                float result = coeff * (Navg / (n * nvr) + energy / nvr);

                atomicAdd(&tile_TVR[ly][lx], result);
            }
        }
    }

    __syncthreads();

    // Write back to global memory
    int gx = tile_origin_x + threadIdx.x;
    int gy = tile_origin_y + threadIdx.y;

    if (gx < N_GRID_X && gy < N_GRID_Y &&
        threadIdx.x < TILE_X && threadIdx.y < TILE_Y) {

        int idx = gx + gy * N_GRID_X;
        atomicAdd(&TVR[idx], tile_TVR[threadIdx.y][threadIdx.x]);
    }
}

void compute_moments(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter){
    int n_particles = N_PARTICLES;
    int num_cells = fc.nx * fc.ny;

    cudaMemcpyToSymbol(kb, &kb_host, sizeof(float));
    cudaMemcpyToSymbol(m, &m_host, sizeof(float));

    fc.setZero();

    dim3 threadsPerBlock2d(TILE_X, TILE_Y);
    dim3 blocksPerGrid2d(
        (N_GRID_X + TILE_X - 1) / TILE_X,
        (N_GRID_Y + TILE_Y - 1) / TILE_Y,
        1  // You can parallelize over z if needed
    );

    //if(Tiling)
    //  deposit_density_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, fc.d_N, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    //else
    //  deposit_density_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, fc.d_N, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    compute_density(sorter);
    cudaDeviceSynchronize();

    // compute bulk velocity (MC)
    //if(Tiling)
    //  deposit_velocity_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, fc.d_N, fc.d_Ux, fc.d_Uy, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    //else
    //  deposit_velocity_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, fc.d_N, pc.d_vx, pc.d_vy, fc.d_Ux, fc.d_Uy, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    deposit_velocity_2d_sorted<<<blocksPerGrid, threadsPerBlock>>>(pc.d_vx, pc.d_vy, sorter.d_cell_offsets, fc.d_Ux, fc.d_Uy, num_cells);
    cudaDeviceSynchronize();

    // compute bulk temperature (MC)
    //if(Tiling)
    //  deposit_temperature_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, fc.d_N, fc.d_Ux, fc.d_Uy, fc.d_T, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    //else
    //  deposit_temperature_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, fc.d_N, pc.d_vx, pc.d_vy, fc.d_Ux, fc.d_Uy, fc.d_T, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    deposit_temperature_2d_sorted<<<blocksPerGrid, threadsPerBlock>>>(pc.d_vx, pc.d_vy, sorter.d_cell_offsets, fc.d_Ux, fc.d_Uy, fc.d_T, num_cells);
    cudaDeviceSynchronize();

    // compute density (VR)
    //if(Tiling)
    //  deposit_density_2d_VR_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_w, fc.d_N, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    //else
    //  deposit_density_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_w, fc.d_N, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    deposit_density_2d_VR_sorted<<<blocksPerGrid, threadsPerBlock>>>(pc.d_w, sorter.d_cell_offsets, fc.d_NVR, num_cells, n_particles);
    cudaDeviceSynchronize();

    // compute velocity (VR)
    //if(Tiling)
    //  deposit_velocity_2d_VR_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    //else
    //  deposit_velocity_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_UxVR, fc.d_UyVR, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    deposit_velocity_2d_VR_sorted<<<blocksPerGrid, threadsPerBlock>>>(pc.d_vx, pc.d_vy, pc.d_w, sorter.d_cell_offsets, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, num_cells);
    cudaDeviceSynchronize();

    // compute temperature (VR)
    //if(Tiling)
    //  deposit_temperature_2d_VR_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_N, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    //else
    //  deposit_temperature_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_N, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    deposit_temperature_2d_VR_sorted<<<blocksPerGrid, threadsPerBlock>>>(pc.d_vx, pc.d_vy, pc.d_w, fc.d_UxVR, fc.d_UyVR, sorter.d_cell_offsets, fc.d_NVR, fc.d_TVR, num_cells);
    cudaDeviceSynchronize();
}

