#include <cstdio>
#include <cmath>
#include <cufft.h>
#include <cuda_runtime.h>
// compile with: nvcc -O3 -arch=sm_70 poisson_fft_solver.cu -lcufft -o poisson_solver

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
__global__ void compute_residual_kernel(const float *phi, const float *N, float *residual,
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

    residual[idx] = lap + N[idx];
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

// ---------------------------------------------
// Main: test the solver
// ---------------------------------------------
int main() {
    int NX = 64, NY = 64;
    float dx = 1.0f/NX, dy = 1.0f/NY;
    int N = NX*NY;

    // Host arrays
    float* h_rhs = new float[N];
    float* h_phi = new float[N];

    // Example RHS: N(x,y) = sin(2πx) sin(2πy)
    for (int j=0; j<NY; j++) {
        for (int i=0; i<NX; i++) {
            float x = i*dx;
            float y = j*dy;
            h_rhs[j*NX + i] = sinf(2*M_PI*x) * sinf(2*M_PI*y);
        }
    }

    // Device arrays
    float *d_rhs, *d_phi, *d_residual;
    CUDA_CHECK(cudaMalloc(&d_rhs, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_phi, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, N*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rhs, h_rhs, N*sizeof(float), cudaMemcpyHostToDevice));

    // Solve Poisson
    poisson_fft_solver(NX, NY, dx, dy, d_rhs, d_phi);

    // Compute residual
    dim3 threads2D(16,16);
    dim3 blocks2D((NX+15)/16, (NY+15)/16);
    compute_residual_kernel<<<blocks2D, threads2D>>>(d_phi, d_rhs, d_residual, NX, NY, dx, dy);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute L2 norm of residual
    float res_norm = compute_l2_norm(d_residual, N);
    printf("Residual L2 norm = %e\n", res_norm);

    // Cleanup
    delete[] h_rhs;
    delete[] h_phi;
    cudaFree(d_rhs);
    cudaFree(d_phi);
    cudaFree(d_residual);

    return 0;
}
