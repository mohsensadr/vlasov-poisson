#include "SortedDepositor.h"

void SortedDepositor::deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter) {
    int n_particles = N_PARTICLES;
    int num_cells = fc.nx * fc.ny;

    deposit_density_2d_sorted<<<blocksPerGrid, threadsPerBlock>>>(sorter.d_cell_counts, sorter.fc->d_N, num_cells);
    cudaDeviceSynchronize();

    deposit_velocity_2d_sorted<<<blocksPerGrid, threadsPerBlock>>>(pc.d_vx, pc.d_vy, sorter.d_cell_offsets, fc.d_Ux, fc.d_Uy, num_cells);
    cudaDeviceSynchronize();

    deposit_temperature_2d_sorted<<<blocksPerGrid, threadsPerBlock>>>(pc.d_vx, pc.d_vy, sorter.d_cell_offsets, fc.d_Ux, fc.d_Uy, fc.d_T, num_cells);
    cudaDeviceSynchronize();

    deposit_density_2d_VR_sorted<<<blocksPerGrid, threadsPerBlock>>>(pc.d_w, sorter.d_cell_offsets, fc.d_NVR, num_cells, n_particles);
    cudaDeviceSynchronize();

    deposit_velocity_2d_VR_sorted<<<blocksPerGrid, threadsPerBlock>>>(pc.d_vx, pc.d_vy, pc.d_w, sorter.d_cell_offsets, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, num_cells);
    cudaDeviceSynchronize();

    deposit_temperature_2d_VR_sorted<<<blocksPerGrid, threadsPerBlock>>>(pc.d_vx, pc.d_vy, pc.d_w, fc.d_UxVR, fc.d_UyVR, sorter.d_cell_offsets, fc.d_NVR, fc.d_TVR, num_cells);
    cudaDeviceSynchronize();
}

__global__ void deposit_density_2d_sorted(
    const int* __restrict__ d_cell_counts,
    float_type* __restrict__ d_N,
    int num_cells
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cells) return;
    d_N[i] = static_cast<float_type>(d_cell_counts[i]);
}

__global__ void deposit_velocity_2d_sorted(
    const float_type* __restrict__ vx,
    const float_type* __restrict__ vy,
    const int* __restrict__ d_cell_offsets,
    float_type* __restrict__ Ux,
    float_type* __restrict__ Uy,
    int num_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    // Get start and end index for this cell
    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1];

    float_type sum_vx = 0.0;
    float_type sum_vy = 0.0;
    int count = end - start;

    // Sum over particles in this cell
    for (int i = start; i < end; i++) {
        sum_vx += vx[i];
        sum_vy += vy[i];
    }

    // Store average velocity (avoid division by zero)
    if (count > 0) {
        Ux[cell] = sum_vx / count;
        Uy[cell] = sum_vy / count;
    } else {
        Ux[cell] = 0.0;
        Uy[cell] = 0.0;
    }
}

// T = ( <(vx-Ux)^2 + (vy-Uy)^2> ) / (2 * kb/m)
__global__ void deposit_temperature_2d_sorted(
    const float_type* __restrict__ vx,
    const float_type* __restrict__ vy,
    const int*   __restrict__ d_cell_offsets, // size: num_cells + 1
    const float_type* __restrict__ Ux,
    const float_type* __restrict__ Uy,
    float_type* T,
    int num_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1]; // exclusive

    float_type ux = Ux[cell];
    float_type uy = Uy[cell];
    float_type temp_sum = 0.0;
    int npart = end - start;

    for (int i = start; i < end; ++i) {
        float_type dvx = vx[i] - ux;
        float_type dvy = vy[i] - uy;
        temp_sum += dvx * dvx + dvy * dvy;
    }

    T[cell] = (npart > 0) ? temp_sum / (2.0f * npart) : 0.0; // kb/m=1 here
}

__global__ void deposit_density_2d_VR_sorted(
    const float_type* __restrict__ w,          // particle weights
    const int*   __restrict__ d_cell_offsets, // per-cell start indices (size num_cells+1)
    float_type* NVR,                           // output: variance-reduced density
    int num_cells,
    int n_particles
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1]; // exclusive
    int npart = end - start;

    float_type Navg = float_type(n_particles) / float_type(num_cells);
    float_type sum = 0.0;

    for (int i = start; i < end; ++i) {
        sum += 1.0 - w[i];
    }

    NVR[cell] = (npart > 0) ? Navg + sum : Navg;
}

__global__ void deposit_velocity_2d_VR_sorted(
    const float_type* __restrict__ vx,
    const float_type* __restrict__ vy,
    const float_type* __restrict__ w,
    const int*   __restrict__ d_cell_offsets, // start indices of particles per cell
    const float_type* __restrict__ NVR,            // number of particles per cell / density
    float_type* UxVR,                              // output: x-velocity per cell
    float_type* UyVR,                              // output: y-velocity per cell
    int num_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1]; // exclusive
    int npart = end - start;

    float_type sum_vx = 0.0;
    float_type sum_vy = 0.0;

    for (int i = start; i < end; ++i) {
        float_type factor = 1.0 - w[i];
        sum_vx += vx[i] * factor;
        sum_vy += vy[i] * factor;
    }

    // Avoid division by zero
    if (npart > 0) {
        UxVR[cell] = sum_vx / NVR[cell];
        UyVR[cell] = sum_vy / NVR[cell];
    } else {
        UxVR[cell] = 0.0;
        UyVR[cell] = 0.0;
    }
}

__global__ void deposit_temperature_2d_VR_sorted(
    const float_type* __restrict__ vx,
    const float_type* __restrict__ vy,
    const float_type* __restrict__ w,
    const float_type* __restrict__ UxVR,
    const float_type* __restrict__ UyVR,
    const int*   __restrict__ d_cell_offsets, // start indices of particles per cell
    const float_type* __restrict__ NVR,            // VR density
    float_type* TVR,                               // output: VR temperature per cell
    int num_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1]; // exclusive
    int npart = end - start;

    if (npart < 1.0) return;

    // Navg for variance reduction
    float_type Navg = 0.0;
    if (num_cells > 0) {
        Navg = float_type(d_cell_offsets[num_cells]) / float_type(num_cells); // total_particles / num_cells
    }

    float_type temp_sum = 0.0;

    for (int i = start; i < end; ++i) {
        float_type dvx = vx[i] - UxVR[cell];
        float_type dvy = vy[i] - UyVR[cell];
        temp_sum += (dvx*dvx + dvy*dvy) * 0.5 * (1.0 - w[i]);
    }

    temp_sum += Navg;// kb/m=1 here

    TVR[cell] = temp_sum / NVR[cell];
}