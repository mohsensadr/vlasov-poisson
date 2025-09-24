#include "TiledDepositor.h"

void TiledDepositor::deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& /*sorter*/) {
    int n_particles = N_PARTICLES;
    dim3 threadsPerBlock2d(TILE_X, TILE_Y);
    dim3 blocksPerGrid2d(
        (N_GRID_X + TILE_X - 1) / TILE_X,
        (N_GRID_Y + TILE_Y - 1) / TILE_Y,
        1
    );

    deposit_density_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, fc.d_N, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    deposit_velocity_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, fc.d_N, fc.d_Ux, fc.d_Uy, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    deposit_temperature_2d_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, fc.d_N, fc.d_Ux, fc.d_Uy, fc.d_T, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    deposit_density_2d_VR_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_w, fc.d_N, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    deposit_velocity_2d_VR_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    deposit_temperature_2d_VR_tiled<<<blocksPerGrid2d, threadsPerBlock2d>>>(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_N, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
}

__global__ void deposit_density_2d_tiled(
    float_type *x, float_type *y, float_type *N,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
) {
    // Each block handles a TILE_X x TILE_Y tile of the grid
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    float_type dx = N_GRID_X / Lx;
    float_type dy = N_GRID_Y / Ly;

    // Shared memory tile
    __shared__ float_type tile_counts[TILE_Y][TILE_X];

    // Initialize shared memory
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_counts[local_y][local_x] = 0.0;
    }
    __syncthreads();

    // Flatten thread index
    int tid = blockIdx.z * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;

    // Stride over particles
    for (int i = tid; i < n_particles; i += gridDim.z * blockDim.x * blockDim.y) {
        float_type px = x[i];
        float_type py = y[i];

        int gx = int(px * dx);
        int gy = int(py * dy);

        // Skip particles outside this tile
        if (gx >= tile_origin_x && gx < tile_origin_x + TILE_X &&
            gy >= tile_origin_y && gy < tile_origin_y + TILE_Y) {
            int lx = gx - tile_origin_x;
            int ly = gy - tile_origin_y;
            atomicAdd(&tile_counts[ly][lx], 1.0);
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

__global__ void deposit_velocity_2d_tiled(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *N, float_type *Ux, float_type *Uy,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
) {
    // Compute origin of the tile in grid space
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    // Grid scale
    float_type dx = N_GRID_X / Lx;
    float_type dy = N_GRID_Y / Ly;

    // Allocate shared memory tile
    __shared__ float_type tile_Ux[TILE_Y][TILE_X];
    __shared__ float_type tile_Uy[TILE_Y][TILE_X];

    // Zero shared memory (by all threads in block)
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_Ux[local_y][local_x] = 0.0;
        tile_Uy[local_y][local_x] = 0.0;
    }
    __syncthreads();

    // Flatten thread ID across block
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int i = thread_id; i < n_particles; i += threads_per_block) {
        float_type px = x[i];
        float_type py = y[i];

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

                float_type nval = N[idx];  // read global N[idx]
                if (nval > 0.0) {
                    float_type vx_scaled = vx[i] / nval;
                    float_type vy_scaled = vy[i] / nval;

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

__global__ void deposit_temperature_2d_tiled(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *N, const float_type *Ux, const float_type *Uy,
    float_type *T,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
) {
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    float_type dx = float_type(N_GRID_X) / Lx;
    float_type dy = float_type(N_GRID_Y) / Ly;

    __shared__ float_type tile_T[TILE_Y][TILE_X];

    // Initialize shared memory
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_T[local_y][local_x] = 0.0;
    }
    __syncthreads();

    // Flatten thread ID across block and z dimension
    int tid = blockIdx.z * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = gridDim.z * blockDim.x * blockDim.y;

    for (int i = tid; i < n_particles; i += total_threads) {
        float_type px = x[i];
        float_type py = y[i];

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

            float_type nval = N[idx];
            if (nval > 0.0) {
                float_type ux = Ux[idx];
                float_type uy = Uy[idx];
                float_type dvx = vx[i] - ux;
                float_type dvy = vy[i] - uy;
                float_type energy = (dvx * dvx + dvy * dvy);
                float_type temp_contrib = energy / nval / (2.0f * kb / m);

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


__global__ void deposit_density_2d_VR_tiled(
    const float_type *x, const float_type *y,
    const float_type *w,
    const float_type *N,
    float_type *NVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
) {
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    float_type dx = float_type(N_GRID_X) / Lx;
    float_type dy = float_type(N_GRID_Y) / Ly;

    float_type Navg = (1.0 * n_particles) / (N_GRID_X * N_GRID_Y);

    __shared__ float_type tile_NVR[TILE_Y][TILE_X];

    // Initialize shared memory
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_NVR[local_y][local_x] = 0.0;
    }
    __syncthreads();

    // Flatten thread ID
    int tid = blockIdx.z * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = gridDim.z * blockDim.x * blockDim.y;

    for (int i = tid; i < n_particles; i += total_threads) {
        float_type px = x[i];
        float_type py = y[i];

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

            float_type nval = N[idx];
            if (nval > 0.0) {
                float_type delta = Navg / nval + 1.0 - w[i];
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


__global__ void deposit_velocity_2d_VR_tiled(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *w,
    const float_type *NVR,
    float_type *UxVR, float_type *UyVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
) {
    // Tile origin in grid coordinates
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    float_type dx = float_type(N_GRID_X) / Lx;
    float_type dy = float_type(N_GRID_Y) / Ly;

    // Shared memory tiles
    __shared__ float_type tile_UxVR[TILE_Y][TILE_X];
    __shared__ float_type tile_UyVR[TILE_Y][TILE_X];

    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    // Initialize shared memory
    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_UxVR[local_y][local_x] = 0.0;
        tile_UyVR[local_y][local_x] = 0.0;
    }
    __syncthreads();

    // Global thread index
    int tid = blockIdx.z * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = gridDim.z * blockDim.x * blockDim.y;

    // Loop over particles
    for (int i = tid; i < n_particles; i += total_threads) {
        float_type px = x[i];
        float_type py = y[i];

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

            float_type nvr = NVR[idx];
            if (nvr > 0.0) {
                float_type weight_factor = (1.0 - w[i]) / nvr;
                float_type vx_scaled = vx[i] * weight_factor;
                float_type vy_scaled = vy[i] * weight_factor;

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

__global__ void deposit_temperature_2d_VR_tiled(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *w,
    const float_type *N,
    const float_type *NVR,
    const float_type *UxVR, const float_type *UyVR,
    float_type *TVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
) {
    int tile_origin_x = blockIdx.x * TILE_X;
    int tile_origin_y = blockIdx.y * TILE_Y;

    float_type dx = float_type(N_GRID_X) / Lx;
    float_type dy = float_type(N_GRID_Y) / Ly;

    // Shared memory tile
    __shared__ float_type tile_TVR[TILE_Y][TILE_X];

    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    if (local_x < TILE_X && local_y < TILE_Y) {
        tile_TVR[local_y][local_x] = 0.0;
    }
    __syncthreads();

    int tid = blockIdx.z * blockDim.x * blockDim.y +
              threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = gridDim.z * blockDim.x * blockDim.y;

    float_type Navg = float_type(n_particles) / (float_type(N_GRID_X) * float_type(N_GRID_Y));
    const float_type coeff = 1.0 / (kb / m); // precompute inverse thermal energy unit

    for (int i = tid; i < n_particles; i += total_threads) {
        float_type px = x[i];
        float_type py = y[i];

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

            float_type n = N[idx];
            float_type nvr = NVR[idx];

            if (n > 0.0 && nvr > 0.0) {
                float_type ux = UxVR[idx];
                float_type uy = UyVR[idx];

                float_type dvx = vx[i] - ux;
                float_type dvy = vy[i] - uy;

                float_type energy = 0.5 * (dvx * dvx + dvy * dvy) * (1.0 - w[i]);

                float_type result = coeff * (Navg / (n * nvr) + energy / nvr);

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
