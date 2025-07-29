#include <curand_kernel.h>
#include <math.h>
#include <iostream>
#include <fstream>

__device__ float pdf(float x, float Lx, float A, float kx) {
    float normalizer_pdf = (A * sinf(Lx * kx) + Lx * kx) / kx ;
    return ( 1.0f + A * cosf(kx * x) ) / normalizer_pdf;
}

__global__ void initialize_particles(float *x, float *y,
                                     float *vx, float *vy,
                                     float Lx, float Ly,
                                     int N, float A=1.0f, float kx=0.6f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState state;
    curand_init(1234ULL, i, 0, &state);

    // --- Sample x from custom PDF using rejection sampling ---
    float x_sample = 0.0f;
    bool accepted = false;
    float f_max = 1.0f + fabsf(A);  // upper bound for rejection sampling

    for (int attempt = 0; attempt < 100 && !accepted; ++attempt) {
        float x_try = Lx * curand_uniform(&state);
        float u = curand_uniform(&state);
        if (u < pdf(x_try, A, kx, Lx) / f_max) {
            x_sample = x_try;
            accepted = true;
        }
    }
    // fallback in case rejection fails
    if (!accepted) {
        x_sample = Lx * curand_uniform(&state);
    }

    // --- Uniform y in domain ---
    float y_sample = Ly * curand_uniform(&state);
    x[i] = x_sample;
    y[i] = y_sample;

    // --- Maxwellian velocity (Box-Muller) ---
    float u1 = curand_uniform(&state);
    float u2 = curand_uniform(&state);
    float u3 = curand_uniform(&state);
    float u4 = curand_uniform(&state);

    float vx_ = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    float vy_ = sqrtf(-2.0f * logf(u3)) * cosf(2.0f * M_PI * u4);

    // Landau perturbation in velocity
    float alpha = 0.01f;
    vx_ += alpha * cosf(2.0f * M_PI * x_sample / Lx);
    vy_ += alpha * cosf(2.0f * M_PI * y_sample / Ly);

    vx[i] = vx_;
    vy[i] = vy_;
}

// assuming initial density has the same Ux, Uy, T as the global equilibrium
__global__ void initialize_weights(float *x, float *y, float *N, float *w,
        int Ntotal, int N_GRID_X, int N_GRID_Y, float Lx, float Ly, float A=1.0f, float kx=0.6f) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Ntotal) return;

    float dx = Lx/N_GRID_X;
    float dy = Ly/N_GRID_Y;

    int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;

    int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;

    int idx = ix + iy * N_GRID_X;
    
    float Nemp = N[idx];

    float Navg = (1.0f*Ntotal) / (1.0f*N_GRID_X*N_GRID_Y);

    float Ntarget = pdf(x[i], A, kx, Lx) * 1.0 / Ly * dx * dy * Ntotal;

    w[i] = (Navg + Nemp - Ntarget) / Nemp;
}

