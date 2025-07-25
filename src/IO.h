#pragma once

#include "IO.h"
#include <cstdio>   // for snprintf

/**
 * @brief Write the density grid to a CSV file.
 * 
 * @param filename Output file name (with path).
 * @param rho Pointer to the flattened 2D density array.
 */
void write_density_to_csv(const char* filename, float* rho);

/**
 * @brief Create "data/" directory (if needed) and save the density grid for the given step.
 * 
 * @param step The current simulation step.
 * @param rho_host Pointer to host memory containing the density data.
 */
void write_output(int step, float* rho_host);
