#include <curand_kernel.h>
#include <math.h>

__device__ float pdf(float x, float A, float k) {
    return 1.0f + A * cosf(k * x);  // Assumes x ∈ [0, Lx]
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
        if (u < pdf(x_try, A, kx) / f_max) {
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
