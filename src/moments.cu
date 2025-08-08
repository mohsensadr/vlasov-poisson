#include "constants.hpp"
#include <iostream>
#include <fstream>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "moments.cuh"
#include "sorting.cuh"

void compute_density(Sorting& sorter) {
    int n_particles = sorter.n_particles;
    int grid_size = sorter.nx*sorter.ny;  // nx * ny

    // Input keys (cell index per particle)
    const thrust::device_vector<int>& d_cell_indices = sorter.d_cell_indices;

    // Prepare temporary keys and counts
    thrust::device_vector<int> unique_cell_ids(grid_size);
    thrust::device_vector<int> counts(grid_size);

    // Constant values of 1 (each particle contributes 1 to its cell)
    thrust::device_vector<int> ones(n_particles, 1);

    // Compute histogram: number of particles per cell
    auto new_end = thrust::reduce_by_key(
        d_cell_indices.begin(), d_cell_indices.end(), // keys: cell indices
        ones.begin(),                                 // values: 1 per particle
        unique_cell_ids.begin(),                      // output keys
        counts.begin()                                // output counts
    );

    int n_unique_cells = new_end.first - unique_cell_ids.begin();

    // Zero out the field density grid first
    thrust::device_ptr<float> d_N_ptr(sorter.fc->d_N);
    thrust::fill(d_N_ptr, d_N_ptr + grid_size, 0.0f);

    // Scatter counts into d_N (convert int count to float)
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(unique_cell_ids.begin(), counts.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(unique_cell_ids.begin() + n_unique_cells, counts.begin() + n_unique_cells)),
        [d_N_ptr] __device__ (thrust::tuple<int, int> t) {
            int cell = thrust::get<0>(t);
            int count = thrust::get<1>(t);
            d_N_ptr[cell] = static_cast<float>(count);
        }
    );
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

    cudaMemcpyToSymbol(kb, &kb_host, sizeof(float));
    cudaMemcpyToSymbol(m, &m_host, sizeof(float));

    fc.setZero();

    dim3 threadsPerBlock2d(TILE_X, TILE_Y);
    dim3 blocksPerGrid2d(
        (N_GRID_X + TILE_X - 1) / TILE_X,
        (N_GRID_Y + TILE_Y - 1) / TILE_Y,
        1  // You can parallelize over z if needed
    );

    compute_density(sorter);

    //if(Tiling)
    //  deposit_density_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, fc.d_N, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    //else
    //  deposit_density_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, fc.d_N, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute bulk velocity (MC)
    if(Tiling)
      deposit_velocity_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, fc.d_N, fc.d_Ux, fc.d_Uy, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    else
      deposit_velocity_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, fc.d_N, pc.d_vx, pc.d_vy, fc.d_Ux, fc.d_Uy, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute bulk temperature (MC)
    if(Tiling)
      deposit_temperature_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, fc.d_N, fc.d_Ux, fc.d_Uy, fc.d_T, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    else
      deposit_temperature_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, fc.d_N, pc.d_vx, pc.d_vy, fc.d_Ux, fc.d_Uy, fc.d_T, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute density (VR)
    if(Tiling)
      deposit_density_2d_VR_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_w, fc.d_N, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    else
      deposit_density_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_w, fc.d_N, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute velocity (VR)
    if(Tiling)
      deposit_velocity_2d_VR_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    else
      deposit_velocity_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_UxVR, fc.d_UyVR, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    // compute temperature (VR)
    if(Tiling)
      deposit_temperature_2d_VR_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_N, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    else
      deposit_temperature_2d_VR<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_N, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();
}

