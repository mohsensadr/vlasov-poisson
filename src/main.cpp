#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>  // C++17
#include <cstdio>      // for snprintf
#include "constants.hpp"
#include "vlasov_poisson.cuh"

// ./main N_GRID_X N_GRID_Y N_PARTICLES DT NSteps Lx Ly Q_OVER_M threadsPerBlock
// ./main 128 128 100000 0.01 100 1.0 1.0 0.5 256

int main(int argc, char** argv) {
    if (argc > 1) N_GRID_X = std::atoi(argv[1]);
    if (argc > 2) N_GRID_Y = std::atoi(argv[2]);
    if (argc > 3) N_PARTICLES = std::atoi(argv[3]);
    if (argc > 4) DT = std::atof(argv[4]);
    if (argc > 5) NSteps = std::atof(argv[5]);
    if (argc > 6) Lx = std::atof(argv[6]);
    if (argc > 7) Ly = std::atof(argv[7]);
    if (argc > 8) Q_OVER_M = std::atof(argv[8]);
    if (argc > 9){
      threadsPerBlock = std::atof(argv[9]);
        if (threadsPerBlock <= 0) {
          std::cout << "Block size must be a positive integer.\n";
        return -1;
      }
    }

    std::cout << "N_GRID_X: " << N_GRID_X << "\n";
    std::cout << "N_GRID_Y: " << N_GRID_Y << "\n";
    std::cout << "N_PARTICLES: " << N_PARTICLES << "\n";
    std::cout << "DT: " << DT << "\n";
    std::cout << "Lx: " << Lx << "\n";
    std::cout << "Ly: " << Ly << "\n";
    std::cout << "Q_OVER_M: " << Q_OVER_M << "\n";
    std::cout << "threadsPerBlock: " << threadsPerBlock << "\n";

    run(N_GRID_X, N_GRID_Y,
            N_PARTICLES,
            DT,
            NSteps,
            Lx,
            Ly,
            Q_OVER_M,
            threadsPerBlock
            );

    return 0;
}
