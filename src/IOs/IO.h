#pragma once

#include "IO.h"
#include <cstdio>
#include <string>
#include <Containers/field_container.cuh>

/**
 * @brief Write a flattened 2D array to a CSV file.
 *
 * @param filename Output file name (with path).
 * @param x Pointer to the flattened array (row-major order).
 */
void write_to_csv(const std::string& filename, float_type* x);

/**
 * @brief Write a field (host array) to the `data/` folder for a given simulation step.
 *
 * The output filename format is: data/<label>_step_XXX.csv
 *
 * @param step The current time step.
 * @param x Pointer to the host data array.
 * @param label Variable label for the output filename.
 */
void write_output(int step, float_type* x, std::string s);

/**
 * @brief Post-process and dump several fields (copied from device to host).
 *
 * Copies fields from the GPU to CPU and writes them to disk.
 *
 * @param fc Reference to the field container holding device pointers.
 * @param step The current simulation step number.
 */
void post_proc(FieldContainer& fc, int step);
