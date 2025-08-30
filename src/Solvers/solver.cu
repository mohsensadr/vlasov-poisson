#include <cuda_runtime.h>
#include <math.h>
#include <cufft.h>

#include "Solvers/solver.cuh"
#include "Constants/constants.hpp"

// ---------------------------------------------
// Utility: check CUDA errors
// ---------------------------------------------
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(-1); \
    }

// ---------------------------------------------
// Kernel: scale spectrum (solve in Fourier space)
// ---------------------------------------------
__global__ void scale_kernel(const cufftComplex* rhs_hat, cufftComplex* phi_hat,
                             int NX, int NY, float dx, float dy) {
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    if (kx >= NX/2+1 || ky >= NY) return;

    int idx = ky * (NX/2+1) + kx;

    float kx_val = 2.0f * M_PI * ((kx <= NX/2) ? kx : kx - NX) / (NX * dx);
    float ky_val = 2.0f * M_PI * ((ky <= NY/2) ? ky : ky - NY) / (NY * dy);

    float denom = (kx_val*kx_val + ky_val*ky_val);

    if (denom > 1e-14f) {
        phi_hat[idx].x = rhs_hat[idx].x / denom;
        phi_hat[idx].y = rhs_hat[idx].y / denom;
    } else {
        // Zero-frequency mode -> set to 0 (fix nullspace)
        phi_hat[idx].x = 0.0f;
        phi_hat[idx].y = 0.0f;
    }
}

// ---------------------------------------------
// Kernel: scale real field (normalize after iFFT)
// ---------------------------------------------
__global__ void scale_real_kernel(float* arr, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) arr[idx] *= alpha;
}

// ---------------------------------------------
// Kernel: compute residual r = Laplacian(phi) + N
// ---------------------------------------------
__global__ void compute_residual_kernel(const float *phi, const float *d_rhs, float *residual,
                                        int NX, int NY, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX || j >= NY) return;

    int ip = (i + 1) % NX;
    int im = (i - 1 + NX) % NX;
    int jp = (j + 1) % NY;
    int jm = (j - 1 + NY) % NY;

    int idx    = j * NX + i;
    int idx_im = j * NX + im;
    int idx_ip = j * NX + ip;
    int idx_jm = jm * NX + i;
    int idx_jp = jp * NX + i;

    float lap = (phi[idx_im] - 2.0f*phi[idx] + phi[idx_ip]) / (dx*dx)
              + (phi[idx_jm] - 2.0f*phi[idx] + phi[idx_jp]) / (dy*dy);

    residual[idx] = lap + d_rhs[idx];
}

// ---------------------------------------------
// Kernel: reduction for L2 norm
// ---------------------------------------------
__global__ void l2_reduce_kernel(const float* vec, float* block_sums, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < N) {
        float v = vec[idx];
        sum += v*v;
    }
    if (idx + blockDim.x < N) {
        float v = vec[idx + blockDim.x];
        sum += v*v;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}

float compute_l2_norm(const float* d_vec, int N, int threads=256) {
    int blocks = (N + threads*2 - 1) / (threads*2);
    float* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(float)));

    l2_reduce_kernel<<<blocks, threads, threads*sizeof(float)>>>(d_vec, d_block_sums, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_block_sums = new float[blocks];
    CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, blocks*sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i=0; i<blocks; i++) sum += h_block_sums[i];

    delete[] h_block_sums;
    cudaFree(d_block_sums);

    return sqrt(sum);
}

// ---------------------------------------------
// Direct Poisson solver using cuFFT
// ---------------------------------------------
void poisson_fft_solver(int NX, int NY, float dx, float dy,
                        float* d_rhs, float* d_phi) {
    cufftHandle planR2C, planC2R;
    cufftPlan2d(&planR2C, NY, NX, CUFFT_R2C);
    cufftPlan2d(&planC2R, NY, NX, CUFFT_C2R);

    int nComplex = NY * (NX/2 + 1);
    cufftComplex* d_rhs_hat;
    cufftComplex* d_phi_hat;
    CUDA_CHECK(cudaMalloc(&d_rhs_hat, sizeof(cufftComplex)*nComplex));
    CUDA_CHECK(cudaMalloc(&d_phi_hat, sizeof(cufftComplex)*nComplex));

    // Forward FFT
    cufftExecR2C(planR2C, (cufftReal*)d_rhs, d_rhs_hat);

    // Solve in spectral space
    dim3 threads2D(16,16);
    dim3 blocks2D((NX/2+1 + 15)/16, (NY+15)/16);
    scale_kernel<<<blocks2D, threads2D>>>(d_rhs_hat, d_phi_hat, NX, NY, dx, dy);

    // Inverse FFT
    cufftExecC2R(planC2R, d_phi_hat, (cufftReal*)d_phi);

    // Normalize
    int size = NX*NY;
    float alpha = 1.0f / float(size);
    scale_real_kernel<<<(size+255)/256, 256>>>(d_phi, alpha, size);

    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
    cudaFree(d_rhs_hat);
    cudaFree(d_phi_hat);
}

static __device__ int periodic_index(int i, int N) {
    return (i + N) % N;
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

__global__ void compute_rhs_kernel(const float* d_N, float* d_rhs, int NX, int NY, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX || j >= NY) return;

    int idx = j * NX + i;
    d_rhs[idx] = - d_N[idx] / (dx * dy);

    // for testing
    //float x = i*dx;
    //float y = j*dy;
    //d_rhs[idx] = sinf(2*M_PI*x) * sinf(2*M_PI*y);

}

void solve_poisson_periodic(FieldContainer& fc) {
    // Compute residual
    // Block and grid config — safe for any N_GRID_X, N_GRID_Y
    int threadsPerBlockX = min(32, (int) (sqrt(threadsPerBlock))); // 32*32=1024 is max thread per block of T40 GPUs
    int threadsPerBlockY = min(32, (int) (sqrt(threadsPerBlock)));
    dim3 blockDim(threadsPerBlockX, threadsPerBlockY);
    dim3 gridDim(
        (N_GRID_X + blockDim.x - 1) / blockDim.x,
        (N_GRID_Y + blockDim.y - 1) / blockDim.y
    );

    int size = N_GRID_X * N_GRID_Y;
    float *d_rhs, *d_phi, *d_residual;

    CUDA_CHECK(cudaMalloc(&d_rhs, size*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_phi, size*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, size*sizeof(float)));

    compute_rhs_kernel<<<gridDim, blockDim>>>(fc.d_N, d_rhs, N_GRID_X, N_GRID_Y, dx, dy);
    cudaDeviceSynchronize();

    float res_normalizer = compute_l2_norm(d_rhs, size); CUDA_CHECK(cudaDeviceSynchronize());

    // Solve Poisson
    poisson_fft_solver(N_GRID_X, N_GRID_Y, fc.dx, fc.dy, d_rhs, d_phi);

    // Compute Residual
    compute_residual_kernel<<<gridDim, blockDim>>>(d_phi, d_rhs, d_residual, N_GRID_X, N_GRID_Y, dx, dy);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute L2 norm of residual
    float res_norm = compute_l2_norm(d_residual, size); CUDA_CHECK(cudaDeviceSynchronize());
    res_norm /= res_normalizer;
    //printf("Residual L2 norm = %e\n", res_norm);

    compute_electric_field_kernel_periodic<<<gridDim, blockDim>>>(d_phi, fc.d_Ex, fc.d_Ey, N_GRID_X, N_GRID_Y, dx, dy);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_phi);
    cudaFree(d_residual);
    cudaFree(d_rhs);
}
