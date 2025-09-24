#pragma once
#include <string>
/**
 * @file constants.hpp
 * @brief Global configuration and physical constants for the Vlasov–Poisson solver.
 *
 * This header declares simulation parameters used across host and device code.
 * Device constants are conditionally declared for CUDA compilation.
 */

 // datatype to be used
#ifndef FLOAT_TYPE
#define FLOAT_TYPE float
#endif

using float_type = FLOAT_TYPE;

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
extern float_type DT;               ///< Time step
extern int NSteps;             ///< Number of simulation steps
extern float_type Lx;               ///< Domain size in X
extern float_type Ly;               ///< Domain size in Y
extern float_type dx;               ///< Grid spacing in X
extern float_type dy;               ///< Grid spacing in Y
extern std::string problem;    ///< string specifying the problem
extern int Nm;                 ///< Number of moments

// ----------------------------
// Physical constants
// ----------------------------
extern float_type Q_OVER_M;    ///< Charge-to-mass ratio (q/m)

// ----------------------------
// CUDA kernel configuration
// ----------------------------
extern int threadsPerBlock; ///< CUDA threads per block
extern int blocksPerGrid;   ///< CUDA blocks per grid

enum class DepositionMode {
    BRUTE,
    TILING,
    SORTING
};

extern DepositionMode depositionMode;

enum class VRMode {
    BASIC,
    MXE
};

extern VRMode vrMode;

enum class RhsMode {
    MC,
    VR
};

extern RhsMode rhsMode;

// ----------------------------
// Host-side constants
// ----------------------------
constexpr float_type kb_host = 1.0; ///< Boltzmann constant on host (normalized units)
constexpr float_type m_host  = 1.0; ///< Particle mass on host (normalized units)

// ----------------------------
// Device-side constants
// ----------------------------
#ifdef __CUDACC__
__constant__ float_type kb;         ///< Boltzmann constant on device
__constant__ float_type m;          ///< Particle mass on device
#endif

#define PI_F 3.14159265358979f
