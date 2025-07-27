#include <cuda_runtime.h>
#include <math.h>
#include <solver.cuh>

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

void solve_poisson_jacobi(float *N_d, float *Ex_d, float *Ey_d,
                   int N_GRID_X, int N_GRID_Y, float dx, float dy, int threadsPerBlock) {
    int size = N_GRID_X * N_GRID_Y;
    size_t bytes = size * sizeof(float);

    float *phi_old, *phi_new;
    cudaMalloc(&phi_old, bytes);
    cudaMalloc(&phi_new, bytes);
    cudaMemset(phi_old, 0, bytes);
    cudaMemset(phi_new, 0, bytes);

    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim((N_GRID_X + threadsPerBlock-1) / threadsPerBlock, (N_GRID_Y + threadsPerBlock-1) / threadsPerBlock);

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        jacobi_iteration_kernel<<<gridDim, blockDim>>>(N_d, phi_new, phi_old, N_GRID_X, N_GRID_Y, dx, dy);
        
        apply_neumann_bc_kernel<<<gridDim, blockDim>>>(phi_new, N_GRID_X, N_GRID_Y);

        float* tmp = phi_old;
        phi_old = phi_new;
        phi_new = tmp;
    }

    // Compute electric field from potential
    compute_electric_field_kernel<<<gridDim, blockDim>>>(phi_old, Ex_d, Ey_d, N_GRID_X, N_GRID_Y, dx, dy);

    // Cleanup
    cudaFree(phi_old);
    cudaFree(phi_new);
}
