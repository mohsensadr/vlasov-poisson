#pragma once

extern int N_GRID_X;
extern int N_GRID_Y;

extern int N_PARTICLES;
extern float DT;
extern int NSteps;
extern float Lx;
extern float Ly;
extern float Q_OVER_M;

extern int threadsPerBlock;

// Host-accessible constants
constexpr float kb_host = 1.0f; //1.380649e-23f;
constexpr float m_host  = 1.0f; //9.1093837e-31f;

#ifdef __CUDACC__
// Device-side constants (only visible to nvcc)
__constant__ float kb;
__constant__ float m;
#endif