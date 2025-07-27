// moments.cuh

#pragma once

void compute_moments(float *d_x, float *d_y, float *d_vx, float *d_vy, 
    float *d_N, float *d_Ux, float *d_Uy, float *d_T,
    int n_particles, int N_GRID_X, int N_GRID_Y, float Lx, float Ly,
    int blocksPerGrid, int threadsPerBlock);