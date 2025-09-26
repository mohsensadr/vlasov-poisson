#include <cuda_runtime.h>
#include <math.h>
#include <cufft.h>

#include "Solvers/solver.cuh"
#include "Constants/constants.hpp"

// ---------------------------------------------
// Utility: check CUDA errors
// ---------------------------------------------
#define CUDA_CHECK(err) \
    if ((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(-1); \
    }

#define CUFFT_CHECK(err) \
    if ((err) != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d (%s:%d)\n", err, __FILE__, __LINE__); \
        exit(-1); \
    }

template<typename T> struct cufft_traits;

template<>
struct cufft_traits<float> {
    using real_t    = cufftReal;
    using complex_t = cufftComplex;
    static constexpr real_t EPS = 1e-7f;
    static cufftResult exec_forward(cufftHandle plan, real_t* in, complex_t* out) {
        return cufftExecR2C(plan, in, out);
    }
    static cufftResult exec_inverse(cufftHandle plan, complex_t* in, real_t* out) {
        return cufftExecC2R(plan, in, out);
    }
};

template<>
struct cufft_traits<double> {
    using real_t    = cufftDoubleReal;
    using complex_t = cufftDoubleComplex;
    static constexpr real_t EPS = 1e-14;
    static cufftResult exec_forward(cufftHandle plan, real_t* in, complex_t* out) {
        return cufftExecD2Z(plan, in, out);
    }
    static cufftResult exec_inverse(cufftHandle plan, complex_t* in, real_t* out) {
        return cufftExecZ2D(plan, in, out);
    }
};

// -------------------------------------------------------------
// Choose precision at compile time
// -------------------------------------------------------------
using traits = cufft_traits<float_type>;
using real_t = typename traits::real_t;
using complex_t = typename traits::complex_t;

// ---------------------------------------------
// Kernel: scale spectrum (solve in Fourier space)
// ---------------------------------------------
__global__ void scale_kernel(const complex_t* rhs_hat, complex_t* phi_hat,
                             int NX, int NY, real_t dx, real_t dy) {
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    if (kx >= NX/2+1 || ky >= NY) return;

    int idx = ky * (NX/2+1) + kx;

    real_t two_pi = (real_t)(2.0 * M_PI);
    real_t kx_val = two_pi * ((kx <= NX/2) ? (real_t)kx : (real_t)(kx - NX)) / ((real_t)NX * dx);
    real_t ky_val = two_pi * ((ky <= NY/2) ? (real_t)ky : (real_t)(ky - NY)) / ((real_t)NY * dy);

    real_t denom = kx_val*kx_val + ky_val*ky_val;

    if (denom > traits::EPS) {
        phi_hat[idx].x = rhs_hat[idx].x / denom;
        phi_hat[idx].y = rhs_hat[idx].y / denom;
    } else {
        phi_hat[idx].x = (real_t)0.0;
        phi_hat[idx].y = (real_t)0.0;
    }
}

// ---------------------------------------------
// Kernel: scale real field (normalize after iFFT)
// ---------------------------------------------
__global__ void scale_real_kernel(real_t* arr, real_t alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) arr[idx] *= alpha;
}

// ---------------------------------------------
// Kernel: compute residual r = Laplacian(phi) + N
// ---------------------------------------------
__global__ void compute_residual_kernel(const real_t *phi, const real_t *d_rhs, real_t *residual,
                                        int NX, int NY, real_t dx, real_t dy) {
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

    real_t lap = (phi[idx_im] - 2.0*phi[idx] + phi[idx_ip]) / (dx*dx)
               + (phi[idx_jm] - 2.0*phi[idx] + phi[idx_jp]) / (dy*dy);

    residual[idx] = lap + d_rhs[idx];
}

// ---------------------------------------------
// Kernel: reduction for L2 norm
// ---------------------------------------------
__global__ void l2_reduce_kernel(const real_t* vec, real_t* block_sums, int N) {
    extern __shared__ real_t sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    real_t sum = (real_t)0.0;
    if (idx < N) {
        real_t v = vec[idx];
        sum += v*v;
    }
    if (idx + blockDim.x < N) {
        real_t v = vec[idx + blockDim.x];
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

real_t compute_l2_norm(const real_t* d_vec, int N, int threads=256) {
    int blocks = (N + threads*2 - 1) / (threads*2);
    real_t* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(real_t)));

    l2_reduce_kernel<<<blocks, threads, threads*sizeof(real_t)>>>(d_vec, d_block_sums, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    real_t* h_block_sums = new real_t[blocks];
    CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, blocks*sizeof(real_t), cudaMemcpyDeviceToHost));

    real_t sum = 0.0;
    for (int i=0; i<blocks; i++) sum += h_block_sums[i];

    delete[] h_block_sums;
    cudaFree(d_block_sums);

    return std::sqrt(sum);
}

// ---------------------------------------------
// Direct Poisson solver using cuFFT
// ---------------------------------------------
void poisson_fft_solver(int NX, int NY, float_type dx, float_type dy,
                        float_type* d_rhs, float_type* d_phi) {
    using traits = cufft_traits<float_type>;
    using real_t = typename traits::real_t;
    using complex_t = typename traits::complex_t;

    cufftHandle planR2C, planC2R;
    cufftPlan2d(&planR2C, NY, NX, (std::is_same<float_type,double>::value) ? CUFFT_D2Z : CUFFT_R2C);
    cufftPlan2d(&planC2R, NY, NX, (std::is_same<float_type,double>::value) ? CUFFT_Z2D : CUFFT_C2R);

    int nComplex = NY * (NX/2 + 1);
    complex_t* d_rhs_hat;
    complex_t* d_phi_hat;
    CUDA_CHECK(cudaMalloc(&d_rhs_hat, sizeof(complex_t)*nComplex));
    CUDA_CHECK(cudaMalloc(&d_phi_hat, sizeof(complex_t)*nComplex));

    // Forward FFT
    CUFFT_CHECK(traits::exec_forward(planR2C, (real_t*)d_rhs, d_rhs_hat));

    // Solve in spectral space
    dim3 threads2D(16,16);
    dim3 blocks2D((NX/2+1 + 15)/16, (NY+15)/16);
    scale_kernel<<<blocks2D, threads2D>>>(d_rhs_hat, d_phi_hat, NX, NY, dx, dy);

    // Inverse FFT
    CUFFT_CHECK(traits::exec_inverse(planC2R, d_phi_hat, (real_t*)d_phi));

    // Normalize
    int size = NX*NY;
    float_type alpha = 1.0 / float_type(size);
    scale_real_kernel<<<(size+255)/256, 256>>>(d_phi, alpha, size);

    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
    cudaFree(d_rhs_hat);
    cudaFree(d_phi_hat);
}

static __device__ int periodic_index(int i, int N) {
    return (i + N) % N;
}

__global__ void compute_electric_field_kernel_periodic(const real_t *phi, real_t *Ex, real_t *Ey,
                                                       int N_GRID_X, int N_GRID_Y, real_t dx, real_t dy) {
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

        Ex[idx] = -(phi[idx_ip] - phi[idx_im]) / ((real_t)2.0 * dx);
        Ey[idx] = -(phi[idx_jp] - phi[idx_jm]) / ((real_t)2.0 * dy);
    }
}

__global__ void compute_rhs_kernel(const real_t* d_N, real_t* d_rhs, int NX, int NY, real_t dx, real_t dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX || j >= NY) return;

    int idx = j * NX + i;
    d_rhs[idx] = d_N[idx] / (dx * dy);

    // Debug example:
    // real_t x = i*dx;
    // real_t y = j*dy;
    // d_rhs[idx] = sin((real_t)(2*M_PI)*x) * sin((real_t)(2*M_PI)*y);
}

void solve_poisson_periodic(FieldContainer& fc) {
    dim3 blockDim(16,16);
    dim3 gridDim(
        (N_GRID_X + blockDim.x - 1) / blockDim.x,
        (N_GRID_Y + blockDim.y - 1) / blockDim.y
    );

    int size = N_GRID_X * N_GRID_Y;
    real_t *d_rhs, *d_residual;

    CUDA_CHECK(cudaMalloc(&d_rhs, size*sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_residual, size*sizeof(real_t)));

    // --------------------------
    // 1. Monte Carlo estimate
    // --------------------------
    compute_rhs_kernel<<<gridDim, blockDim>>>(fc.d_N, d_rhs, N_GRID_X, N_GRID_Y, dx, dy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    real_t res_normalizer = compute_l2_norm(d_rhs, size);

    poisson_fft_solver(N_GRID_X, N_GRID_Y, fc.dx, fc.dy, d_rhs, fc.d_phi);

    compute_residual_kernel<<<gridDim, blockDim>>>(fc.d_phi, d_rhs, d_residual, N_GRID_X, N_GRID_Y, dx, dy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    real_t res_norm = compute_l2_norm(d_residual, size) / res_normalizer;

    compute_electric_field_kernel_periodic<<<gridDim, blockDim>>>(fc.d_phi, fc.d_Ex, fc.d_Ey, N_GRID_X, N_GRID_Y, dx, dy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------
    // 2. Variance-Reduced estimate
    // --------------------------
    compute_rhs_kernel<<<gridDim, blockDim>>>(fc.d_NVR, d_rhs, N_GRID_X, N_GRID_Y, dx, dy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    res_normalizer = compute_l2_norm(d_rhs, size);

    poisson_fft_solver(N_GRID_X, N_GRID_Y, fc.dx, fc.dy, d_rhs, fc.d_phiVR);

    compute_residual_kernel<<<gridDim, blockDim>>>(fc.d_phiVR, d_rhs, d_residual, N_GRID_X, N_GRID_Y, dx, dy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    res_norm = compute_l2_norm(d_residual, size) / res_normalizer;

    compute_electric_field_kernel_periodic<<<gridDim, blockDim>>>(fc.d_phiVR, fc.d_ExVR, fc.d_EyVR, N_GRID_X, N_GRID_Y, dx, dy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    cudaFree(d_residual);
    cudaFree(d_rhs);
}