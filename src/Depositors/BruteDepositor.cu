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

__global__ void deposit_density_2d(float *x, float *y, float *N, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&N[idx], 1.0f);
    }
}

/*
__global__ void deposit_velocity_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&Ux[idx], vx[i]/N[idx]);
        atomicAdd(&Uy[idx], vy[i]/N[idx]);
    }
}
*/

// Pass 1: accumulate raw sums of velocity
__global__ void deposit_velocity_accumulate(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *N,
    float *Ux, float *Uy,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
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
    float *Ux, float *Uy,
    const float *N,
    int num_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        if (N[idx] > 0.0f) {
            Ux[idx] /= N[idx];
            Uy[idx] /= N[idx];
        } else {
            Ux[idx] = 0.0f;
            Uy[idx] = 0.0f;
        }
    }
}

// Host-side wrapper that does both steps
inline void deposit_velocity_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
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

/*
__global__ void deposit_temperature_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, float *T, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        float energy = (vx[i]-Ux[idx])*(vx[i]-Ux[idx]);
        energy += (vy[i]-Uy[idx])*(vy[i]-Uy[idx]);
        atomicAdd(&T[idx], energy/N[idx]/(2.0f*kb/m));
    }
}
*/

// Pass 1: accumulate raw kinetic energy (per cell)
__global__ void deposit_temperature_accumulate(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *Ux, const float *Uy,
    float *T,          // stores accumulated energy (temporary)
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;

        // (vx - Ux)^2 + (vy - Uy)^2
        float dvx = vx[i] - Ux[idx];
        float dvy = vy[i] - Uy[idx];
        float energy = dvx * dvx + dvy * dvy;

        atomicAdd(&T[idx], energy);
    }
}

// Pass 2: normalize energy into temperature
__global__ void deposit_temperature_finalize(
    float *T,
    const float *N,
    int num_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        if (N[idx] > 0.0f) {
            // divide by particle count and constants
            T[idx] = T[idx] / N[idx] / 2.0f;
            // ignore division by (kb / m), as it's set to 1 here.
        } else {
            T[idx] = 0.0f;
        }
    }
}

// Host wrapper
inline void deposit_temperature_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, float *T, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
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

/*
__global__ void deposit_density_2d_VR(float *x, float *y, float *w, float *N, float *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        float Navg = (1.0f*n_particles) / (1.0f*N_GRID_X*N_GRID_Y);
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&NVR[idx], Navg/N[idx] + 1.0f - w[i] );
    }
}
*/

// Pass 1: accumulate raw VR contributions
__global__ void deposit_density_VR_accumulate(
    const float *x, const float *y,
    const float *w,
    float *NVR,         // accumulator
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;

        atomicAdd(&NVR[idx], (1.0f - w[i]));
    }
}

__global__ void deposit_density_VR_finalize(
    float *NVR,
    float Navg,
    int num_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
          NVR[idx] += Navg;
    }
}

// Host wrapper
inline void deposit_density_2d_VR(float *x, float *y, float *w, float *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  ) {
    float Navg = (1.0f * n_particles) / (1.0f * N_GRID_X * N_GRID_Y);

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

/*
__global__ void deposit_velocity_2d_VR(float *x, float *y, float *vx, float*vy, float *w,
            float *UxVR, float *UyVR, float *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&UxVR[idx], vx[i] * ( 1.0f - w[i] ) / NVR[idx] );
        atomicAdd(&UyVR[idx], vy[i] * ( 1.0f - w[i] ) / NVR[idx] );
    }
}
*/

// Pass 1: accumulate raw weighted momentum
__global__ void deposit_velocity_VR_accumulate(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *w,
    float *UxVR, float *UyVR,   // accumulators
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;

        // Accumulate momentum contributions
        atomicAdd(&UxVR[idx], vx[i] * (1.0f - w[i]));
        atomicAdd(&UyVR[idx], vy[i] * (1.0f - w[i]));
    }
}

// Pass 2: finalize by dividing by NVR
__global__ void deposit_velocity_VR_finalize(
    float *UxVR, float *UyVR,
    const float *NVR,
    int num_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        float denom = NVR[idx];
        if (denom > 0.0f) {
            UxVR[idx] /= denom;
            UyVR[idx] /= denom;
        } else {
            UxVR[idx] = 0.0f;
            UyVR[idx] = 0.0f;
        }
    }
}

// Host wrapper
inline void deposit_velocity_2d_VR(float *x, float *y, float *vx, float*vy, float *w,
            float *UxVR, float *UyVR, float *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly) {
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

/*
__global__ void deposit_temperature_2d_VR(float *x, float *y, float *vx, float *vy, float *w, float *N, float *NVR, float *UxVR, float *UyVR, float *TVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        float Navg = (1.0f*n_particles) / (1.0f*N_GRID_X*N_GRID_Y);
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        float energy = (vx[i]-UxVR[idx])*(vx[i]-UxVR[idx]);
        energy += (vy[i]-UyVR[idx])*(vy[i]-UyVR[idx]);
        float ans = Navg/(kb/m)/NVR[idx]/N[idx] + ( energy*(1.0f-w[i])/2.0f ) / (kb/m) / NVR[idx];
        atomicAdd(&TVR[idx], ans);
    }
}
*/

// Pass 1: accumulate raw energy contributions
__global__ void deposit_temperature_VR_accumulate(
    const float *x, const float *y,
    const float *vx, const float *vy,
    const float *w,
    const float *UxVR, const float *UyVR,
    float *TVR,
    int n_particles,
    int N_GRID_X, int N_GRID_Y,
    float Lx, float Ly
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;

        float dvx = vx[i] - UxVR[idx];
        float dvy = vy[i] - UyVR[idx];
        float energy = 0.5f * (dvx*dvx + dvy*dvy) * (1.0f - w[i]); // accumulate energy weighted by (1-w)

        atomicAdd(&TVR[idx], energy);
    }
}

// Pass 2: finalize per-cell temperature
__global__ void deposit_temperature_VR_finalize(
    float *TVR,
    const float *NVR,
    const float *N,
    float Navg,
    int num_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        float denom = NVR[idx];
        if (N[idx] > 1.0f) {
            TVR[idx] = (1.0f * Navg + TVR[idx])/ denom;
            // n0=Navg, T0 = 1
            // also, kb/m = 1
        } else {
            TVR[idx] = 1.0f;
        }
    }
}

// Host wrapper
inline void deposit_temperature_2d_VR(float *x, float *y, float *vx, float *vy, float *w, float *N, float *NVR, float *UxVR, float *UyVR, float *TVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  ) {
    float Navg = (1.0f * n_particles) / (1.0f * N_GRID_X * N_GRID_Y);

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
