// IO.cpp
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

void write_density_to_csv(const char *filename, float *rho) {
    std::ofstream out(filename);
    for (int j = 0; j < N_GRID_Y; ++j) {
        for (int i = 0; i < N_GRID_X; ++i) {
            out << rho[i + j * N_GRID_X];
            if (i < N_GRID_X - 1) out << ",";
        }
        out << "\n";
    }
    out.close();
}

void write_output(int step, float* rho_host) {
    fs::create_directories("data");

    char filename[64];
    snprintf(filename, sizeof(filename), "data/rho_step_%03d.csv", step);
    write_density_to_csv(filename, rho_host);  // assumed declared somewhere
    std::cout << "Saved: " << filename << "\n";
}
