#ifndef SOLVER_H
#define SOLVER_H

#include <cuda_runtime.h>
#include <Containers/field_container.cuh>

// Main solver function
void solve_poisson_periodic(FieldContainer& fc);

#endif  // SOLVER_H
