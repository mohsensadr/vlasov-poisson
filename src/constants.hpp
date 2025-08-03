#pragma once
#include <string>
/**
 * @file constants.hpp
 * @brief Global configuration and physical constants for the Vlasov–Poisson solver.
 *
 * This header declares simulation parameters used across host and device code.
 * Device constants are conditionally declared for CUDA compilation.
 */

// ----------------------------
// Grid and domain configuration
// ----------------------------
extern int N_GRID_X;      ///< Grid resolution in X direction
extern int N_GRID_Y;      ///< Grid resolution in Y direction
extern int grid_size;     ///< Total number of grid cells (N_GRID_X * N_GRID_Y)

// ----------------------------
// Particle simulation details
// ----------------------------
extern int N_PARTICLES;        ///< Number of particles
extern float DT;               ///< Time step
extern int NSteps;             ///< Number of simulation steps
extern float Lx;               ///< Domain size in X
extern float Ly;               ///< Domain size in Y
extern float dx;               ///< Grid spacing in X
extern float dy;               ///< Grid spacing in Y
extern std::string problem;    ///< string specifying the problem

// ----------------------------
// Physical constants
// ----------------------------
extern float Q_OVER_M;    ///< Charge-to-mass ratio (q/m)

// ----------------------------
// CUDA kernel configuration
// ----------------------------
extern int threadsPerBlock; ///< CUDA threads per block
extern int blocksPerGrid;   ///< CUDA blocks per grid
extern bool Tiling;         ///< CUDA boolean for doing tiling or not

// ----------------------------
// Host-side constants
// ----------------------------
constexpr float kb_host = 1.0f; ///< Boltzmann constant on host (normalized units)
constexpr float m_host  = 1.0f; ///< Particle mass on host (normalized units)

// ----------------------------
// Device-side constants
// ----------------------------
#ifdef __CUDACC__
__constant__ float kb;         ///< Boltzmann constant on device
__constant__ float m;          ///< Particle mass on device
#endif

#define PI_F 3.14159265358979f
