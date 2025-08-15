#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

#include "constants.hpp"
#include "vlasov_poisson.cuh"
// Using simple CUDA-compatible PDF approach

// ./main N_GRID_X N_GRID_Y N_PARTICLES DT NSteps Lx Ly threadsPerBlock deposition_mode VRMode [pdf_type] [pdf_params...]
// deposition_mode: brute | tiling | sorting
// VRMode: basic | MXE
// Examples:
// ./main 128 128 1000000 0.01 100 1.0 1.0 256 brute basic gaussian 0.5
// ./main 128 128 1000000 0.01 100 1.0 1.0 256 tiling MXE cosine 1.0 0.5
// ./main 128 128 1000000 0.01 100 1.0 1.0 256 sorting basic double_gaussian 0.1 0.2 0.3 0.4 0.7 0.6 0.5 0.5

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

    if (argc > 9) {
        std::string mode_arg(argv[9]);
        std::transform(mode_arg.begin(), mode_arg.end(), mode_arg.begin(), ::tolower);

        if (mode_arg == "brute") {
            depositionMode = DepositionMode::BRUTE;
        } else if (mode_arg == "tiling") {
            depositionMode = DepositionMode::TILING;
        } else if (mode_arg == "sorting") {
            depositionMode = DepositionMode::SORTING;
        } else {
            std::cerr << "Unknown deposition mode: " << mode_arg
                      << "\nValid options: brute, tiling, sorting\n";
            return -1;
        }
    } else {
        depositionMode = DepositionMode::BRUTE; // default
    }

    if (argc > 10) {
        std::string mode_arg(argv[10]);
        std::transform(mode_arg.begin(), mode_arg.end(), mode_arg.begin(), ::tolower);

        if (mode_arg == "basic") {
            vrMode = VRMode::BASIC;
        } else if (mode_arg == "mxe") {
            vrMode = VRMode::MXE;
        } else {
            std::cerr << "Unknown VR mode: " << mode_arg
                      << "\nValid options: basic, mxe\n";
            return -1;
        }
    } else {
        vrMode = VRMode::BASIC; // default
    }

    // PDF selection
    std::string pdf_type = "gaussian";  // default
    float pdf_params[8] = {0., 0., 0., 0., 0., 0., 0., 0.};
    
    if (argc > 11) {
        pdf_type = argv[11];
        
        // Parse PDF parameters
        for (int i = 12; i < argc; ++i) {
            pdf_params[i-12] = std::atof(argv[i]);
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
    std::cout << "Running with mode: ";
    switch (depositionMode) {
        case DepositionMode::BRUTE:   std::cout << "BRUTE\n"; break;
        case DepositionMode::TILING:  std::cout << "TILING\n"; break;
        case DepositionMode::SORTING: std::cout << "SORTING\n"; break;
    }
    std::cout << "Running with VR mode: ";
        switch (vrMode) {
        case VRMode::BASIC:   std::cout << "BASIC\n"; break;
        case VRMode::MXE:  std::cout << "MXE\n"; break;
    }
    std::cout << "PDF Type: " << pdf_type << "\n";
    std::cout << "PDF Parameters: ";
    for (float param : pdf_params) {
        std::cout << param << " ";
    }
    std::cout << "\n";

    auto start_time = std::chrono::high_resolution_clock::now();

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

    // Make sure GPU has finished all work before stopping timer
    cudaDeviceSynchronize();

    std::cout << "Execution time: "
          << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count()
          << " seconds" << std::endl;

    return 0;
}
