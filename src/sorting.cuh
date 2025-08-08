// sorting.cuh
#pragma once

#include <thrust/device_vector.h>
#include "particle_container.cuh"
#include "field_container.cuh"

class Sorting {
private:
    ParticleContainer* pc;
    FieldContainer* fc;

    int nx, ny;
    float xmin, ymin;
    float dx, dy;
    int n_particles;

    thrust::device_vector<int> d_cell_indices;
    thrust::device_vector<int> d_particle_indices;

public:
    Sorting(ParticleContainer& pc_, FieldContainer& fc_);

    void sort_particles_by_cell();

private:
    void reorder_particle_data();

};

// Kernel declaration
__global__ void compute_cell_indices(
    float* d_x, float* d_y,
    int* d_cell_indices,
    int n,
    int nx, int ny,
    float xmin, float ymin,
    float dx, float dy
);
