#include <curand_kernel.h>
#include <math.h>
#include <iostream>
#include <fstream>

__device__ float pdf(float x, float y, float Lx, float Ly, float A, float kx) {
    float normalizer_pdf = (A * sinf(Lx * kx) + Lx * kx) / kx ;
    return ( 1.0f + A * cosf(kx * x) ) / normalizer_pdf * 1.0f / Ly;
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
    float x_sample = 0.0f, y_sample = 0.0f;
    bool accepted = false;

    float normalizer_pdf = (A * sinf(Lx * kx) + Lx * kx) / kx;
    float p_max = (1.0f + A) / normalizer_pdf * (1.0f / Ly);

    for (int attempt = 0; attempt < 200 && !accepted; ++attempt) {
        float x_try = Lx * curand_uniform(&state);
        float y_try = Ly * curand_uniform(&state);
        float u = curand_uniform(&state);  // u in [0, 1)

        float p = pdf(x_try, y_try, Lx, Ly, A, kx);
        if (u < p / p_max) {
            x_sample = x_try;
            y_sample = y_try;
            accepted = true;
        }
    }
    // fallback in case rejection fails
    if (!accepted) {
        x_sample = Lx * curand_uniform(&state);
        y_sample = Ly * curand_uniform(&state);
    }

    // Write accepted sample
    x[i] = x_sample;
    y[i] = y_sample;

    // --- Normal velocity
    vx[i] = curand_normal(&state);
    vy[i] = curand_normal(&state);
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

    float Ntarget = pdf(x[i], y[i], Lx, Ly, A, kx) * dx * dy * Ntotal;

    w[i] = (Navg + Nemp - Ntarget) / Nemp;
}

