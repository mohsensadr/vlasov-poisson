#ifndef PARTICLE_INITIALIZER_H
#define PARTICLE_INITIALIZER_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * @brief Kernel to initialize particle positions and velocities using rejection sampling
 *        for a given probability density function (PDF).
 *
 * @tparam PDF Callable struct or functor representing the PDF
 * @param x Device array for particle x positions
 * @param y Device array for particle y positions
 * @param vx Device array for particle x velocities
 * @param vy Device array for particle y velocities
 * @param Lx Domain length in x
 * @param Ly Domain length in y
 * @param N Number of particles
 * @param pdf PDF functor to use for rejection sampling
 */
template<typename PDF>
__global__ void initialize_particles(float_type *x, float_type *y,
                                     float_type *vx, float_type *vy,
                                     float_type Lx, float_type Ly,
                                     int N, PDF pdf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState state;
    curand_init(1234ULL, i, 0, &state);

    float_type x_sample = 0.0, y_sample = 0.0;
    bool accepted = false;

    float_type p_max = pdf.pmax();

    for (int attempt = 0; attempt < 1000 && !accepted; ++attempt) {
        float_type x_try = Lx * curand_uniform(&state);
        float_type y_try = Ly * curand_uniform(&state);
        float_type u = curand_uniform(&state);
        float_type p = pdf(x_try, y_try);
        if (u < p / p_max) {
            x_sample = x_try;
            y_sample = y_try;
            accepted = true;
        }
    }

    if (!accepted) {
        printf("sampler failed!");
        x_sample = Lx * curand_uniform(&state);
        y_sample = Ly * curand_uniform(&state);
    }

    x[i] = x_sample;
    y[i] = y_sample;

    vx[i] = curand_normal(&state);
    vy[i] = curand_normal(&state);
}

/**
 * @brief Kernel to initialize particle weights to adjust local empirical density to match a target PDF.
 *
 * @tparam PDF Callable struct or functor representing the PDF
 * @param x Device array of particle x positions
 * @param y Device array of particle y positions
 * @param N Device array containing local particle counts per cell
 * @param w Device array for output weights
 * @param Ntotal Total number of particles
 * @param N_GRID_X Number of grid cells in x
 * @param N_GRID_Y Number of grid cells in y
 * @param Lx Domain length in x
 * @param Ly Domain length in y
 * @param pdf PDF functor to use for target density
 */
template<typename PDF>
__global__ void initialize_weights(float_type *x, float_type *y, float_type *N, float_type *w,
                                   int Ntotal, int N_GRID_X, int N_GRID_Y,
                                   float_type Lx, float_type Ly, PDF pdf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Ntotal) return;

    float_type dx = Lx / N_GRID_X;
    float_type dy = Ly / N_GRID_Y;

    int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
    int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
    int idx = ix + iy * N_GRID_X;

    float_type Nemp = N[idx];
    float_type Navg = float_type(Ntotal) / (N_GRID_X * N_GRID_Y);
    float_type Ntarget = pdf(x[i], y[i]) * dx * dy * Lx * Ly * Ntotal;

    w[i] = (Navg + Nemp - Ntarget) / Nemp;
}

#endif  // PARTICLE_INITIALIZER_H
