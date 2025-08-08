// sorting.cu
#include "sorting.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>

Sorting::Sorting(ParticleContainer& pc_, FieldContainer& fc_)
    : pc(&pc_), fc(&fc_), n_particles(pc_.n_particles),
      nx(fc_.nx), ny(fc_.ny),
      xmin(fc_.xmin), ymin(fc_.ymin),
      dx(fc_.dx), dy(fc_.dy)
{
    d_cell_indices.resize(n_particles);
    d_particle_indices.resize(n_particles);
}

void Sorting::sort_particles_by_cell() {
    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;

    compute_cell_indices<<<blocks, threads>>>(
        pc->d_x, pc->d_y,
        thrust::raw_pointer_cast(d_cell_indices.data()),
        n_particles,
        nx, ny,
        xmin, ymin,
        dx, dy
    );
    cudaDeviceSynchronize();

    thrust::sequence(d_particle_indices.begin(), d_particle_indices.end());

    thrust::sort_by_key(
        d_cell_indices.begin(), d_cell_indices.end(),
        d_particle_indices.begin()
    );

    reorder_particle_data();
}

void Sorting::reorder_particle_data() {
    thrust::device_ptr<float> x_ptr(pc->d_x);
    thrust::device_ptr<float> y_ptr(pc->d_y);
    thrust::device_ptr<float> vx_ptr(pc->d_vx);
    thrust::device_ptr<float> vy_ptr(pc->d_vy);
    thrust::device_ptr<float> w_ptr(pc->d_w);

    thrust::device_vector<float> x_sorted(n_particles);
    thrust::device_vector<float> y_sorted(n_particles);
    thrust::device_vector<float> vx_sorted(n_particles);
    thrust::device_vector<float> vy_sorted(n_particles);
    thrust::device_vector<float> w_sorted(n_particles);

    thrust::gather(d_particle_indices.begin(), d_particle_indices.end(), x_ptr, x_sorted.begin());
    thrust::gather(d_particle_indices.begin(), d_particle_indices.end(), y_ptr, y_sorted.begin());
    thrust::gather(d_particle_indices.begin(), d_particle_indices.end(), vx_ptr, vx_sorted.begin());
    thrust::gather(d_particle_indices.begin(), d_particle_indices.end(), vy_ptr, vy_sorted.begin());
    thrust::gather(d_particle_indices.begin(), d_particle_indices.end(), w_ptr, w_sorted.begin());

    thrust::copy(x_sorted.begin(), x_sorted.end(), x_ptr);
    thrust::copy(y_sorted.begin(), y_sorted.end(), y_ptr);
    thrust::copy(vx_sorted.begin(), vx_sorted.end(), vx_ptr);
    thrust::copy(vy_sorted.begin(), vy_sorted.end(), vy_ptr);
    thrust::copy(w_sorted.begin(), w_sorted.end(), w_ptr);
}

__global__ void compute_cell_indices(
    float* d_x, float* d_y,
    int* d_cell_indices,
    int n,
    int nx, int ny,
    float xmin, float ymin,
    float dx, float dy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int ix = (d_x[i] - xmin) / dx;
    int iy = (d_y[i] - ymin) / dy;

    ix = max(0, min(ix, nx - 1));
    iy = max(0, min(iy, ny - 1));

    d_cell_indices[i] = iy * nx + ix;
}
