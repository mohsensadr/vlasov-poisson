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
float_type DT = 0.01;          ///< Time step size
int NSteps = 100;          ///< Number of simulation steps
int Nm = 3;                ///< Number of moments

// Domain dimensions and spacing
float_type Lx = 1.0;                       ///< Domain length in X
float_type Ly = 1.0;                       ///< Domain length in Y
float_type dx = 0.01;                      ///< Grid spacing in X
float_type dy = 0.01;                      ///< Grid spacing in Y
std::string problem = "LandauDamping"; ///< string specifying the problem

// Physical constants
float_type Q_OVER_M = 0.001;     ///< Charge-to-mass ratio (q/m)

// CUDA kernel launch configuration
int threadsPerBlock = 256; ///< CUDA threads per block
int blocksPerGrid = 256;   ///< CUDA blocks per grid
DepositionMode depositionMode;
VRMode vrMode;
RhsMode rhsMode;