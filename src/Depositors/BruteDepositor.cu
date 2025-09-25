#include "BruteDepositor.h"

void BruteDepositor::deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& /*sorter*/) {
    int n_particles = N_PARTICLES;

    deposit_density_2d<<<blocksPerGrid, threadsPerBlock>>>(pc.d_x, pc.d_y, fc.d_N, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    deposit_velocity_2d(pc.d_x, pc.d_y, fc.d_N, pc.d_vx, pc.d_vy, fc.d_Ux, fc.d_Uy, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    deposit_temperature_2d(pc.d_x, pc.d_y, fc.d_N, pc.d_vx, pc.d_vy, fc.d_Ux, fc.d_Uy, fc.d_T, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    deposit_density_2d_VR(pc.d_x, pc.d_y, pc.d_w, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    deposit_velocity_2d_VR(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_UxVR, fc.d_UyVR, fc.d_NVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();

    deposit_temperature_2d_VR(pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, pc.d_w, fc.d_N, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
    cudaDeviceSynchronize();
}

__global__ void deposit_density_2d(float_type *x, float_type *y, float_type *N, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&N[idx], 1.0);
    }
}

// Pass 1: accumulate raw sums of velocity
__global__ void deposit_velocity_accumulate(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *N,
    float_type *Ux, float_type *Uy,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;

        atomicAdd(&Ux[idx], vx[i]);
        atomicAdd(&Uy[idx], vy[i]);
    }
}

// Pass 2: finalize by dividing by N[idx]
__global__ void deposit_velocity_finalize(
    float_type *Ux, float_type *Uy,
    const float_type *N,
    int num_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        if (N[idx] > 0.0) {
            Ux[idx] /= N[idx];
            Uy[idx] /= N[idx];
        } else {
            Ux[idx] = 0.0;
            Uy[idx] = 0.0;
        }
    }
}

// Host-side wrapper that does both steps
inline void deposit_velocity_2d(float_type *x, float_type *y, float_type *N, float_type *vx, float_type *vy, float_type *Ux, float_type *Uy, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly
    ) {
    int num_cells = N_GRID_X*N_GRID_Y;

    // Step 1: accumulate sums
    deposit_velocity_accumulate<<<blocksPerGrid, threadsPerBlock>>>(
        x, y, vx, vy, N, Ux, Uy,
        n_particles, N_GRID_X, N_GRID_Y, Lx, Ly
    );

    // Step 2: finalize averages
    deposit_velocity_finalize<<<blocksPerGrid, threadsPerBlock>>>(
        Ux, Uy, N, num_cells
    );
}

// Pass 1: accumulate raw kinetic energy (per cell)
__global__ void deposit_temperature_accumulate(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *Ux, const float_type *Uy,
    float_type *T,          // stores accumulated energy (temporary)
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;

        // (vx - Ux)^2 + (vy - Uy)^2
        float_type dvx = vx[i] - Ux[idx];
        float_type dvy = vy[i] - Uy[idx];
        float_type energy = dvx * dvx + dvy * dvy;

        atomicAdd(&T[idx], energy);
    }
}

// Pass 2: normalize energy into temperature
__global__ void deposit_temperature_finalize(
    float_type *T,
    const float_type *N,
    int num_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        if (N[idx] > 0.0) {
            // divide by particle count and constants
            T[idx] = T[idx] / N[idx] / 2.0;
            // ignore division by (kb / m), as it's set to 1 here.
        } else {
            T[idx] = 0.0;
        }
    }
}

// Host wrapper
inline void deposit_temperature_2d(float_type *x, float_type *y, float_type *N, float_type *vx, float_type *vy, float_type *Ux, float_type *Uy, float_type *T, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly
  ) {
    int num_cells = N_GRID_X*N_GRID_Y;

    // Step 1: accumulate raw energy
    deposit_temperature_accumulate<<<blocksPerGrid, threadsPerBlock>>>(
        x, y, vx, vy, Ux, Uy,
        T,
        n_particles, N_GRID_X, N_GRID_Y, Lx, Ly
    );

    // Step 2: normalize into actual temperature
    deposit_temperature_finalize<<<blocksPerGrid, threadsPerBlock>>>(
        T, N, num_cells
    );
}

// Pass 1: accumulate raw VR contributions
__global__ void deposit_density_VR_accumulate(
    const float_type *x, const float_type *y,
    const float_type *w,
    float_type *NVR,         // accumulator
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;

        atomicAdd(&NVR[idx], (1.0 - w[i]));
    }
}

__global__ void deposit_density_VR_finalize(
    float_type *NVR,
    float_type Navg,
    int num_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
          NVR[idx] += Navg;
    }
}

// Host wrapper
inline void deposit_density_2d_VR(float_type *x, float_type *y, float_type *w, float_type *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly
  ) {
    float_type Navg = (1.0 * n_particles) / (1.0 * N_GRID_X * N_GRID_Y);

    // Step 1: accumulate contributions
    deposit_density_VR_accumulate<<<blocksPerGrid, threadsPerBlock>>>(
        x, y, w, NVR,
        n_particles,
        N_GRID_X, N_GRID_Y,
        Lx, Ly
    );

    // Step 2: finalize
    deposit_density_VR_finalize<<<blocksPerGrid, threadsPerBlock>>>(NVR, Navg, N_GRID_X * N_GRID_Y);
}

// Pass 1: accumulate raw weighted momentum
__global__ void deposit_velocity_VR_accumulate(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *w,
    float_type *UxVR, float_type *UyVR,   // accumulators
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;

        // Accumulate momentum contributions
        atomicAdd(&UxVR[idx], vx[i] * (1.0 - w[i]));
        atomicAdd(&UyVR[idx], vy[i] * (1.0 - w[i]));
    }
}

// Pass 2: finalize by dividing by NVR
__global__ void deposit_velocity_VR_finalize(
    float_type *UxVR, float_type *UyVR,
    const float_type *NVR,
    int num_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        float_type denom = NVR[idx];
        if (denom > 0.0) {
            UxVR[idx] /= denom;
            UyVR[idx] /= denom;
        } else {
            UxVR[idx] = 0.0;
            UyVR[idx] = 0.0;
        }
    }
}

// Host wrapper
inline void deposit_velocity_2d_VR(float_type *x, float_type *y, float_type *vx, float_type *vy, float_type *w,
            float_type *UxVR, float_type *UyVR, float_type *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly) {
    // Step 1: accumulate momentum contributions
    deposit_velocity_VR_accumulate<<<blocksPerGrid, threadsPerBlock>>>(
        x, y, vx, vy, w,
        UxVR, UyVR,
        n_particles, N_GRID_X, N_GRID_Y, Lx, Ly
    );

    // Step 2: normalize per-cell
    deposit_velocity_VR_finalize<<<blocksPerGrid, threadsPerBlock>>>(
        UxVR, UyVR, NVR, N_GRID_X * N_GRID_Y
    );
}

// Pass 1: accumulate raw energy contributions
__global__ void deposit_temperature_VR_accumulate(
    const float_type *x, const float_type *y,
    const float_type *vx, const float_type *vy,
    const float_type *w,
    const float_type *UxVR, const float_type *UyVR,
    float_type *TVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float_type Lx, float_type Ly
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;

        float_type dvx = vx[i] - UxVR[idx];
        float_type dvy = vy[i] - UyVR[idx];
        float_type energy = 0.5 * (dvx*dvx + dvy*dvy) * (1.0 - w[i]); // accumulate energy weighted by (1-w)

        atomicAdd(&TVR[idx], energy);
    }
}

// Pass 2: finalize per-cell temperature
__global__ void deposit_temperature_VR_finalize(
    float_type *TVR,
    const float_type *NVR,
    const float_type *N,
    float_type Navg,
    int num_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        float_type denom = NVR[idx];
        if (N[idx] > 1.0) {
            TVR[idx] = (1.0 * Navg + TVR[idx])/ denom;
            // n0=Navg, T0 = 1
            // also, kb/m = 1
        } else {
            TVR[idx] = 1.0;
        }
    }
}

// Host wrapper
inline void deposit_temperature_2d_VR(float_type *x, float_type *y, float_type *vx, float_type *vy, float_type *w, float_type *N, float_type *NVR, float_type *UxVR, float_type *UyVR, float_type *TVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly
  ) {
    float_type Navg = (1.0 * n_particles) / (1.0 * N_GRID_X * N_GRID_Y);

    // Step 1: accumulate energy
    deposit_temperature_VR_accumulate<<<blocksPerGrid, threadsPerBlock>>>(
        x, y, vx, vy, w,
        UxVR, UyVR, TVR,
        n_particles, N_GRID_X, N_GRID_Y, Lx, Ly
    );

    // Step 2: finalize temperature
    deposit_temperature_VR_finalize<<<blocksPerGrid, threadsPerBlock>>>(
        TVR, NVR, N, Navg, N_GRID_X * N_GRID_Y
    );
}
