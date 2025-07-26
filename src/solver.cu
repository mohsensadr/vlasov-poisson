#include <cuda_runtime.h>
#include <math.h>
#include <solver.cuh>

__global__ void jacobi_iteration_kernel(const float *rho, float *phi_new, const float *phi_old,
                                        int N_GRID_X, int N_GRID_Y, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N_GRID_X - 1 && j > 0 && j < N_GRID_Y - 1) {
        int idx = j * N_GRID_X + i;
        phi_new[idx] = 1.0f/(2.0f*dx*dx + dy*dy) * (
            + (phi_old[idx - 1] + phi_old[idx + 1]) * dy*dy               // left + right
            + (phi_old[idx - N_GRID_X] + phi_old[idx + N_GRID_X]) * dx*dx // bottom + top
            + rho[idx] * dx*dx * dy*dy );
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

void solve_poisson_jacobi(float *rho_d, float *Ex_d, float *Ey_d,
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
        jacobi_iteration_kernel<<<gridDim, blockDim>>>(rho_d, phi_new, phi_old, N_GRID_X, N_GRID_Y, dx, dy);
        
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
