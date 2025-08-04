#include <cuda_runtime.h>
#include <math.h>
#include <solver.cuh>
#include "constants.hpp"

static __device__ int periodic_index(int i, int N) {
    return (i + N) % N;
}

__global__ void apply_neumann_bc_kernel(float *phi, int N_GRID_X, int N_GRID_Y) {
    int idx;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bottom and top boundaries (j = 0 and j = N_GRID_Y-1)
    if (i < N_GRID_X) {
        // Bottom: phi[i, 0] = phi[i, 1]
        idx = 0 * N_GRID_X + i;
        phi[idx] = phi[idx + N_GRID_X];

        // Top: phi[i, N-1] = phi[i, N-2]
        idx = (N_GRID_Y - 1) * N_GRID_X + i;
        phi[idx] = phi[idx - N_GRID_X];
    }

    // Left and right boundaries (i = 0 and i = N_GRID_X-1)
    if (i < N_GRID_Y) {
        // Left: phi[0, j] = phi[1, j]
        idx = i * N_GRID_X + 0;
        phi[idx] = phi[idx + 1];

        // Right: phi[N-1, j] = phi[N-2, j]
        idx = i * N_GRID_X + (N_GRID_X - 1);
        phi[idx] = phi[idx - 1];
    }
}

__global__ void jacobi_iteration_kernel(const float *N, float *phi_new, const float *phi_old,
                                        int N_GRID_X, int N_GRID_Y, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N_GRID_X - 1 && j > 0 && j < N_GRID_Y - 1) {
        int idx = j * N_GRID_X + i;
        phi_new[idx] = 1.0f/(2.0f*dx*dx + dy*dy) * (
            + (phi_old[idx - 1] + phi_old[idx + 1]) * dy*dy               // left + right
            + (phi_old[idx - N_GRID_X] + phi_old[idx + N_GRID_X]) * dx*dx // bottom + top
            + N[idx] * dx * dy ); // Number of particles [-] / volume [dx*dy] * dx^2*dy^2
    }
}

__global__ void jacobi_iteration_kernel_periodic(const float *N, float *phi_new, const float *phi_old,
                                        int N_GRID_X, int N_GRID_Y, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N_GRID_X && j < N_GRID_Y) {
        int ip = periodic_index(i + 1, N_GRID_X);
        int im = periodic_index(i - 1, N_GRID_X);
        int jp = periodic_index(j + 1, N_GRID_Y);
        int jm = periodic_index(j - 1, N_GRID_Y);

        int idx = j * N_GRID_X + i;
        int idx_im = j * N_GRID_X + im;
        int idx_ip = j * N_GRID_X + ip;
        int idx_jm = jm * N_GRID_X + i;
        int idx_jp = jp * N_GRID_X + i;

        phi_new[idx] = 1.0f / (2.0f * (dx*dx + dy*dy)) * (
            (phi_old[idx_im] + phi_old[idx_ip]) * dy * dy +
            (phi_old[idx_jm] + phi_old[idx_jp]) * dx * dx +
            - N[idx] * dx * dy
        );
    }
}

__global__ void jacobi_iteration_kernel_periodic_tiled(
    const float *N, float *phi_new, const float *phi_old,
    int N_GRID_X, int N_GRID_Y, float dx, float dy)
{
    __shared__ float tile[TILE_Y + 2][TILE_X + 2];

    int global_i = blockIdx.x * TILE_X + threadIdx.x;
    int global_j = blockIdx.y * TILE_Y + threadIdx.y;

    int local_i = threadIdx.x + 1; // +1 for halo
    int local_j = threadIdx.y + 1;

    // Only load interior cell if in bounds
    if (global_i < N_GRID_X && global_j < N_GRID_Y)
        tile[local_j][local_i] = phi_old[global_j * N_GRID_X + global_i];

    // Periodic boundary indices for halo loading
    int i_left   = (global_i - 1 + N_GRID_X) % N_GRID_X;
    int i_right  = (global_i + 1) % N_GRID_X;
    int j_up     = (global_j - 1 + N_GRID_Y) % N_GRID_Y;
    int j_down   = (global_j + 1) % N_GRID_Y;

    // Load halo columns
    if (threadIdx.x == 0 && global_i < N_GRID_X && global_j < N_GRID_Y)
        tile[local_j][0] = phi_old[global_j * N_GRID_X + i_left];
    if (threadIdx.x == TILE_X - 1 && global_i < N_GRID_X && global_j < N_GRID_Y)
        tile[local_j][TILE_X + 1] = phi_old[global_j * N_GRID_X + i_right];

    // Load halo rows
    if (threadIdx.y == 0 && global_i < N_GRID_X && global_j < N_GRID_Y)
        tile[0][local_i] = phi_old[j_up * N_GRID_X + global_i];
    if (threadIdx.y == TILE_Y - 1 && global_i < N_GRID_X && global_j < N_GRID_Y)
        tile[TILE_Y + 1][local_i] = phi_old[j_down * N_GRID_X + global_i];

    // Load corners
    if (threadIdx.x == 0 && threadIdx.y == 0 && global_i < N_GRID_X && global_j < N_GRID_Y)
        tile[0][0] = phi_old[j_up * N_GRID_X + i_left];
    if (threadIdx.x == TILE_X - 1 && threadIdx.y == 0 && global_i < N_GRID_X && global_j < N_GRID_Y)
        tile[0][TILE_X + 1] = phi_old[j_up * N_GRID_X + i_right];
    if (threadIdx.x == 0 && threadIdx.y == TILE_Y - 1 && global_i < N_GRID_X && global_j < N_GRID_Y)
        tile[TILE_Y + 1][0] = phi_old[j_down * N_GRID_X + i_left];
    if (threadIdx.x == TILE_X - 1 && threadIdx.y == TILE_Y - 1 && global_i < N_GRID_X && global_j < N_GRID_Y)
        tile[TILE_Y + 1][TILE_X + 1] = phi_old[j_down * N_GRID_X + i_right];

    __syncthreads();

    if (global_i < N_GRID_X && global_j < N_GRID_Y) {
        float dx2 = dx * dx;
        float dy2 = dy * dy;
        float denom = 1.0f / (2.0f * (dx2 + dy2));

        int idx = global_j * N_GRID_X + global_i;

        phi_new[idx] = denom * (
            (tile[local_j][local_i - 1] + tile[local_j][local_i + 1]) * dy2 +
            (tile[local_j - 1][local_i] + tile[local_j + 1][local_i]) * dx2 -
            N[idx] * dx * dy
        );
    }
}

__global__ void compute_electric_field_kernel(const float *phi, float *Ex, float *Ey,
                                              int N_GRID_X, int N_GRID_Y, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N_GRID_X - 1 && j > 0 && j < N_GRID_Y - 1) {
        int idx = j * N_GRID_X + i;

        // Central differences for electric field (E = -grad(phi))
        Ex[idx] = -(phi[idx + 1] - phi[idx - 1]) / (2.0f*dx);
        Ey[idx] = -(phi[idx + N_GRID_X] - phi[idx - N_GRID_X]) / (2.0f*dy);
    }
}

__global__ void compute_electric_field_kernel_periodic(const float *phi, float *Ex, float *Ey,
                                              int N_GRID_X, int N_GRID_Y, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N_GRID_X && j < N_GRID_Y) {
        int ip = periodic_index(i + 1, N_GRID_X);
        int im = periodic_index(i - 1, N_GRID_X);
        int jp = periodic_index(j + 1, N_GRID_Y);
        int jm = periodic_index(j - 1, N_GRID_Y);

        int idx = j * N_GRID_X + i;
        int idx_ip = j * N_GRID_X + ip;
        int idx_im = j * N_GRID_X + im;
        int idx_jp = jp * N_GRID_X + i;
        int idx_jm = jm * N_GRID_X + i;

        Ex[idx] = -(phi[idx_ip] - phi[idx_im]) / (2.0f * dx);
        Ey[idx] = -(phi[idx_jp] - phi[idx_jm]) / (2.0f * dy);
    }
}

void solve_poisson_jacobi(FieldContainer& fc) {
    int size = N_GRID_X * N_GRID_Y;
    size_t bytes = size * sizeof(float);

    float *phi_old, *phi_new;
    cudaMalloc(&phi_old, bytes);
    cudaMalloc(&phi_new, bytes);
    cudaMemset(phi_old, 0, bytes);
    cudaMemset(phi_new, 0, bytes);

    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim((N_GRID_X + threadsPerBlock-1) / threadsPerBlock, (N_GRID_Y + threadsPerBlock-1) / threadsPerBlock);

    dim3 blockDim2d(TILE_X, TILE_Y);
    dim3 gridDim2d((N_GRID_X + TILE_X - 1) / TILE_X, 
                    (N_GRID_Y + TILE_Y - 1) / TILE_Y);

    for (int iter = 0; iter < MAX_ITERS; ++iter) {

        if(Tiling)
          jacobi_iteration_kernel_periodic_tiled<<<gridDim2d, blockDim2d>>>(fc.d_N, phi_new, phi_old, N_GRID_X, N_GRID_Y, dx, dy);
        else
          jacobi_iteration_kernel_periodic<<<gridDim, blockDim>>>(fc.d_N, phi_new, phi_old, N_GRID_X, N_GRID_Y, dx, dy);
        
        float* tmp = phi_old;
        phi_old = phi_new;
        phi_new = tmp;
    }

    // Compute electric field from potential
    compute_electric_field_kernel_periodic<<<gridDim, blockDim>>>(phi_old, fc.d_Ex, fc.d_Ey, N_GRID_X, N_GRID_Y, dx, dy);

    // Cleanup
    cudaFree(phi_old);
    cudaFree(phi_new);
}
