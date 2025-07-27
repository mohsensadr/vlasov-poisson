// IO.cpp
#include <filesystem>
#include <iostream>
#include <fstream>   // REQUIRED for std::ofstream
#include <cstdio>    // for snprintf
#include <string>

#include "IO.h"
#include "constants.hpp"
#include <cuda_runtime.h>

namespace fs = std::filesystem;

void write_to_csv(const char *filename, float *x) {
    std::ofstream out(filename);
    for (int j = 0; j < N_GRID_Y; ++j) {
        for (int i = 0; i < N_GRID_X; ++i) {
            out << x[i + j * N_GRID_X];
            if (i < N_GRID_X - 1) out << ",";
        }
        out << "\n";
    }
    out.close();
}

void write_output(int step, float* x, std::string s) {
    fs::create_directories("data");
    char filename[64];
    snprintf(filename, sizeof(filename), "data/%s_step_%03d.csv", s.c_str(), step);
    write_to_csv(filename, x);  // assumed declared somewhere
}

void post_proc(float *d_N, float *d_NVR, float *d_Ux, float *d_Uy, float *d_T, int grid_size, int step){

    float *h_var = new float[grid_size];

    cudaMemcpy(h_var, d_N, sizeof(float) * grid_size, cudaMemcpyDeviceToHost);
    write_output(step, h_var, "N");

    cudaMemcpy(h_var, d_NVR, sizeof(float) * grid_size, cudaMemcpyDeviceToHost);
    write_output(step, h_var, "NVR");

    cudaMemcpy(h_var, d_Ux, sizeof(float) * grid_size, cudaMemcpyDeviceToHost);
    write_output(step, h_var, "Ux");

    cudaMemcpy(h_var, d_Uy, sizeof(float) * grid_size, cudaMemcpyDeviceToHost);
    write_output(step, h_var, "Uy");

    cudaMemcpy(h_var, d_T, sizeof(float) * grid_size, cudaMemcpyDeviceToHost);
    write_output(step, h_var, "T");

    delete[] h_var;

    std::cout << "Wrote postproc in step: " << step << std::endl;
}
