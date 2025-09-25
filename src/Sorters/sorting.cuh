// sorting.cuh
#pragma once

#include <cuda_runtime.h>
#include "Containers/particle_container.cuh"
#include "Containers/field_container.cuh"

class Sorting {
public:
    ParticleContainer* pc;
    FieldContainer* fc;

    int nx, ny;
    int n_particles;
    float_type xmin, ymin;
    float_type dx, dy;

    // device buffers
    int *d_cell_idx = nullptr;           // per-particle cell index (n_particles)
    int *d_cell_counts = nullptr;        // per-cell counts (nx*ny)
    int *d_cell_offsets = nullptr;       // per-cell exclusive prefix sum offsets (nx*ny+1)
    int *d_cell_counters = nullptr;      // temp per-cell counters for scatter (nx*ny)

    // temporary sorted arrays
    float_type *d_x_sorted = nullptr;
    float_type *d_y_sorted = nullptr;
    float_type *d_vx_sorted = nullptr;
    float_type *d_vy_sorted = nullptr;
    float_type *d_w_sorted = nullptr;
    float_type *d_wold_sorted = nullptr;

    Sorting(ParticleContainer& pc_, FieldContainer& fc_);
    ~Sorting();

    // compute cell indices, build histogram, sort particles
    void sort_particles_by_cell(cudaStream_t stream = 0);

private:
    // disable copy
    Sorting(const Sorting&) = delete;
    Sorting& operator=(const Sorting&) = delete;
};
