// IO.cpp
#include <filesystem>
#include <iostream>
#include <fstream>   // REQUIRED for std::ofstream
#include <cstdio>    // for snprintf
#include <string>

#include "IO.h"
#include "constants.hpp"

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

void write_output(int step, float* x, string s) {
    fs::create_directories("data");

    char filename[64];
    snprintf(filename, sizeof(filename), "data/%s_step_%03d.csv", s, step);
    write_density_to_csv(filename, x);  // assumed declared somewhere
    std::cout << "Saved: " << filename << "\n";
}
