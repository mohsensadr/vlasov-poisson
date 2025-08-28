#ifndef SOLVER_H
#define SOLVER_H

#include <cuda_runtime.h>
#include <Containers/field_container.cuh>

// Constants
#define MAX_ITERS 10000
#define TOL 1e-5f

// CUDA kernel declarations
__global__ void jacobi_iteration_kernel(const float *rho, float *phi_new, const float *phi_old,
                                        int N_GRID_X, int N_GRID_Y);

__global__ void compute_electric_field_kernel(const float *phi, float *Ex, float *Ey,
                                              int N_GRID_X, int N_GRID_Y, float dx, float dy);

// Main solver function
void solve_poisson_jacobi(FieldContainer &fc);

#endif  // SOLVER_H
