/**
 * @file constants.cpp
 * @brief Defines global simulation constants and configuration parameters.
 * 
 * These constants are used throughout the Vlasov–Poisson solver, including
 * grid layout, particle parameters, and CUDA kernel launch settings.
 */

#include "constants.hpp"

// Grid configuration
int N_GRID_X = 100;                  ///< Number of grid points in X direction
int N_GRID_Y = 100;                  ///< Number of grid points in Y direction
int grid_size = N_GRID_X * N_GRID_Y; ///< Total number of grid points

// Particle and time integration parameters
int N_PARTICLES = 100000;  ///< Total number of simulation particles
float DT = 0.01f;          ///< Time step size
int NSteps = 100;          ///< Number of simulation steps

// Domain dimensions and spacing
float Lx = 1.0f;                       ///< Domain length in X
float Ly = 1.0f;                       ///< Domain length in Y
float dx = 0.01f;                      ///< Grid spacing in X
float dy = 0.01f;                      ///< Grid spacing in Y
std::string problem = "LandauDamping"; ///< string specifying the problem

// Physical constants
float Q_OVER_M = 1.0f;     ///< Charge-to-mass ratio (q/m)

// CUDA kernel launch configuration
int threadsPerBlock = 256; ///< CUDA threads per block
int blocksPerGrid = 256;   ///< CUDA blocks per grid
bool Tiling = true;       ///< Perform tiling in kernels or not
