#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>  // C++17
#include <cstdio>      // for snprintf
#include <vector>
#include <string>
#include "constants.hpp"
#include "vlasov_poisson.cuh"
// Using simple CUDA-compatible PDF approach

// ./main N_GRID_X N_GRID_Y N_PARTICLES DT NSteps Lx Ly threadsPerBlock [pdf_type] [pdf_params...]
// Examples:
// ./main 128 128 100000 0.01 100 1.0 1.0 256 gaussian 0.1
// ./main 128 128 100000 0.01 100 1.0 1.0 256 cosine 1.0 0.5
// ./main 128 128 100000 0.01 100 1.0 1.0 256 uniform
// ./main 128 128 100000 0.01 100 1.0 1.0 256 double_gaussian 0.1 0.2 0.3 0.4 0.7 0.6 0.5 0.5

int main(int argc, char** argv) {
    if (argc > 1) N_GRID_X = std::atoi(argv[1]);
    if (argc > 2) N_GRID_Y = std::atoi(argv[2]);
    if (argc > 3) N_PARTICLES = std::atoi(argv[3]);
    if (argc > 4) DT = std::atof(argv[4]);
    if (argc > 5) NSteps = std::atof(argv[5]);
    if (argc > 6) Lx = std::atof(argv[6]);
    if (argc > 7) Ly = std::atof(argv[7]);
    if (argc > 8){
      threadsPerBlock = std::atoi(argv[8]);
        if (threadsPerBlock <= 0) {
          std::cout << "Block size must be a positive integer.\n";
        return -1;
      }
    }
    blocksPerGrid = (N_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;

    // PDF selection
    std::string pdf_type = "gaussian";  // default
    float pdf_params[8];
    
    if (argc > 9) {
        pdf_type = argv[9];
        
        // Parse PDF parameters
        for (int i = 10; i < argc; ++i) {
            pdf_params[i-10] = std::atof(argv[i]);
        }
    }

    std::cout << "N_GRID_X: " << N_GRID_X << "\n";
    std::cout << "N_GRID_Y: " << N_GRID_Y << "\n";
    std::cout << "N_PARTICLES: " << N_PARTICLES << "\n";
    std::cout << "DT: " << DT << "\n";
    std::cout << "NSteps: " << NSteps << "\n";
    std::cout << "Lx: " << Lx << "\n";
    std::cout << "Ly: " << Ly << "\n";
    std::cout << "threadsPerBlock: " << threadsPerBlock << "\n";
    std::cout << "blocksPerGrid: " << blocksPerGrid << "\n";
    std::cout << "PDF Type: " << pdf_type << "\n";
    std::cout << "PDF Parameters: ";
    for (float param : pdf_params) {
        std::cout << param << " ";
    }
    std::cout << "\n";

    try {
        // Run simulation with specified PDF
        run(pdf_type, pdf_params);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cout << "Available PDF types:\n";
        std::cout << "  gaussian <variance>\n";
        std::cout << "  cosine <amplitude> <wavenumber>\n";
        std::cout << "  uniform\n";
        std::cout << "  double_gaussian <var1> <var2> <x1> <y1> <x2> <y2> <weight1> <weight2>\n";
        return -1;
    }

    return 0;
}
