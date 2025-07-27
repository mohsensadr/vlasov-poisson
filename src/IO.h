#pragma once

#include "IO.h"
#include <cstdio>   // for snprintf
#include <string>

/**
 * @brief Write the density grid to a CSV file.
 * 
 * @param filename Output file name (with path).
 * @param rho Pointer to the flattened 2D density array.
 */
void write_to_csv(const char* filename, float* x);

/**
 * @brief Create "data/" directory (if needed) and save the density grid for the given step.
 * 
 * @param step The current simulation step.
 * @param rho_host Pointer to host memory containing the density data.
 */
void write_output(int step, float* x, std::string s);

void post_proc(float *d_N, float *d_Ux, float *d_Uy, float *d_T, int grid_size, int step);
