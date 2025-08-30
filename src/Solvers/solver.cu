#include <cuda_runtime.h>
#include <math.h>
#include "Solvers/solver.cuh"
#include "Constants/constants.hpp"

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

    if (i >= N_GRID_X || j >= N_GRID_Y) return;

    int ip = periodic_index(i + 1, N_GRID_X);
    int im = periodic_index(i - 1, N_GRID_X);
    int jp = periodic_index(j + 1, N_GRID_Y);
    int jm = periodic_index(j - 1, N_GRID_Y);

    int idx    = j * N_GRID_X + i;
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

__global__ void compute_l2_norm_kernel(const float* phi_new, const float* phi_old, float* block_sums, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Compute squared differences and load into shared memory
    float sum = 0.0f;
    if (idx < N) {
        float diff = phi_new[idx] - phi_old[idx];
        sum += diff * diff;
    }
    if (idx + blockDim.x < N) {
        float diff = phi_new[idx + blockDim.x] - phi_old[idx + blockDim.x];
        sum += diff * diff;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write block sum to global memory
    if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}


float compute_l2_norm(const float* phi_new, const float* phi_old, int N) {
    int threads = threadsPerBlock;
    int blocks = blocksPerGrid;

    float* d_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(float));

    // Launch fused kernel
    compute_l2_norm_kernel<<<blocks, threads, threads * sizeof(float)>>>(phi_new, phi_old, d_block_sums, N);
    cudaDeviceSynchronize();

    // Reduce remaining blocks on host (or launch kernel again if N is huge)
    float* h_block_sums = new float[blocks];
    cudaMemcpy(h_block_sums, d_block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < blocks; ++i) sum += h_block_sums[i];

    delete[] h_block_sums;
    cudaFree(d_block_sums);

    return sqrt(sum);
}


void solve_poisson_jacobi(FieldContainer& fc) {
    int size = N_GRID_X * N_GRID_Y;
    size_t bytes = size * sizeof(float);

    float *phi_old, *phi_new;
    cudaMalloc(&phi_old, bytes);
    cudaMalloc(&phi_new, bytes);
    cudaMemset(phi_old, 0, bytes);
    cudaMemset(phi_new, 0, bytes);

    // Block and grid config — safe for any N_GRID_X, N_GRID_Y
    int threadsPerBlockX = min(32, (int) (sqrt(threadsPerBlock))); // 32*32=1024 is max thread per block of T40 GPUs
    int threadsPerBlockY = min(32, (int) (sqrt(threadsPerBlock)));
    dim3 blockDim(threadsPerBlockX, threadsPerBlockY);
    dim3 gridDim(
        (N_GRID_X + blockDim.x - 1) / blockDim.x,
        (N_GRID_Y + blockDim.y - 1) / blockDim.y
    );

    float threshold = 1e-5;
    float l2_norm = 1.0f;
    int iter = 0;

    while (iter < MAX_ITERS && l2_norm > threshold) {
        jacobi_iteration_kernel_periodic<<<gridDim, blockDim>>>(fc.d_N, phi_new, phi_old, N_GRID_X, N_GRID_Y, dx, dy);
        cudaDeviceSynchronize();

        l2_norm = compute_l2_norm(phi_new, phi_old, N_GRID_X * N_GRID_Y);

        float* tmp = phi_old;
        phi_old = phi_new;
        phi_new = tmp;

        iter++;
    }

    // Compute electric field from potential
    compute_electric_field_kernel_periodic<<<gridDim, blockDim>>>(phi_old, fc.d_Ex, fc.d_Ey, N_GRID_X, N_GRID_Y, dx, dy);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(phi_old);
    cudaFree(phi_new);
}
