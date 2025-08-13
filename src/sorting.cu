// sorting.cu
#include "sorting.cuh"
#include <cstdio>
#include <cub/cub.cuh> // CUB for DeviceScan::ExclusiveSum

// kernels --------------------------------------------------------------------

__global__ void compute_cell_indices_kernel(
    const float* x, const float* y,
    int* cell_idx,
    int n_particles,
    int nx, int ny,
    float xmin, float ymin,
    float dx, float dy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    int ix = (int)((x[i] - xmin) / dx);
    int iy = (int)((y[i] - ymin) / dy);
    ix = max(0, min(nx - 1, ix));
    iy = max(0, min(ny - 1, iy));
    cell_idx[i] = ix + iy * nx;
}

__global__ void histogram_kernel(
    const int* cell_idx,
    int* cell_counts,
    int n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    int c = cell_idx[i];
    atomicAdd(&cell_counts[c], 1);
}

__global__ void scatter_particles_kernel(
    const float* x, const float* y,
    const float* vx, const float* vy, const float* w,
    const int* cell_idx,
    const int* cell_offsets,
    int* cell_counters,
    float* x_sorted, float* y_sorted,
    float* vx_sorted, float* vy_sorted, float* w_sorted,
    int n_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    int c = cell_idx[i];
    int pos = atomicAdd(&cell_counters[c], 1);
    int dst = cell_offsets[c] + pos;
    x_sorted[dst]  = x[i];
    y_sorted[dst]  = y[i];
    vx_sorted[dst] = vx[i];
    vy_sorted[dst] = vy[i];
    w_sorted[dst]  = w[i];
}

// ---------------------------------------------------------------------------
// Sorting ctor/dtor + helper
// ---------------------------------------------------------------------------

Sorting::Sorting(ParticleContainer& pc_, FieldContainer& fc_)
    : pc(&pc_), fc(&fc_)
{
    n_particles = pc_.n_particles;
    nx = fc->nx;
    ny = fc->ny;

    // Try to extract grid geometry. Adapt if your FieldContainer uses different names.
    // If your FieldContainer stores Lx, Ly instead of dx/dy and xmin/ymin, adjust accordingly.
    xmin = fc->xmin;  // <- ensure FieldContainer has these members
    ymin = fc->ymin;
    dx   = fc->dx;
    dy   = fc->dy;

    size_t np = (size_t)n_particles;
    size_t nc = (size_t)nx * (size_t)ny;

    // allocate per-particle and per-cell arrays
    cudaMalloc(&d_cell_idx,    np * sizeof(int));
    cudaMalloc(&d_cell_counts, nc * sizeof(int));
    cudaMalloc(&d_cell_offsets, (nc+1) * sizeof(int));
    cudaMalloc(&d_cell_counters,nc * sizeof(int));

    // allocate sorted arrays (same size as particle arrays)
    cudaMalloc(&d_x_sorted,  np * sizeof(float));
    cudaMalloc(&d_y_sorted,  np * sizeof(float));
    cudaMalloc(&d_vx_sorted, np * sizeof(float));
    cudaMalloc(&d_vy_sorted, np * sizeof(float));
    cudaMalloc(&d_w_sorted,  np * sizeof(float));
}

Sorting::~Sorting() {
    cudaFree(d_cell_idx);
    cudaFree(d_cell_counts);
    cudaFree(d_cell_offsets);
    cudaFree(d_cell_counters);

    cudaFree(d_x_sorted);
    cudaFree(d_y_sorted);
    cudaFree(d_vx_sorted);
    cudaFree(d_vy_sorted);
    cudaFree(d_w_sorted);
}

// ---------------------------------------------------------------------------
// Public function: tie everything together
// ---------------------------------------------------------------------------

void Sorting::sort_particles_by_cell(cudaStream_t stream) {
    const int TPB = 256;
    int blocks_p = (n_particles + TPB - 1) / TPB;
    int num_cells = nx * ny;

    // 1) compute cell indices
    compute_cell_indices_kernel<<<blocks_p, TPB, 0, stream>>>(
        pc->d_x, pc->d_y,
        d_cell_idx,
        n_particles,
        nx, ny,
        xmin, ymin,
        dx, dy
    );

    // 2) zero cell_counts
    cudaMemsetAsync(d_cell_counts, 0, num_cells * sizeof(int), stream);

    // 3) histogram (atomicAdd)
    histogram_kernel<<<blocks_p, TPB, 0, stream>>>(
        d_cell_idx, d_cell_counts, n_particles
    );

    // synchronize to ensure counts are ready before scan
    cudaStreamSynchronize(stream);

    // 4) exclusive scan (CUB) - cell_offsets = exclusive_scan(cell_counts)
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // determine temporary storage size
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_cell_counts, d_cell_offsets, num_cells);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // run scan
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_cell_counts, d_cell_offsets, num_cells);
    cudaFree(d_temp_storage);

    cudaMemcpy(d_cell_offsets + num_cells, &n_particles, sizeof(int), cudaMemcpyHostToDevice);

    // 5) zero cell_counters (for atomicAdd positions)
    cudaMemsetAsync(d_cell_counters, 0, num_cells * sizeof(int), stream);

    // 6) scatter particles into sorted arrays
    scatter_particles_kernel<<<blocks_p, TPB, 0, stream>>>(
        pc->d_x, pc->d_y, pc->d_vx, pc->d_vy, pc->d_w,
        d_cell_idx,
        d_cell_offsets,
        d_cell_counters,
        d_x_sorted, d_y_sorted,
        d_vx_sorted, d_vy_sorted, d_w_sorted,
        n_particles
    );

    // 7) copy sorted arrays back into ParticleContainer (device-to-device copy)
    cudaMemcpyAsync(pc->d_x, d_x_sorted,  n_particles * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(pc->d_y, d_y_sorted,  n_particles * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(pc->d_vx, d_vx_sorted, n_particles * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(pc->d_vy, d_vy_sorted, n_particles * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(pc->d_w, d_w_sorted,  n_particles * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // ensure work completed
    cudaStreamSynchronize(stream);
}