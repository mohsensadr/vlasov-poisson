#include "constants.hpp"

int N_GRID_X = 100;
int N_GRID_Y = 100;
int grid_size = 100*100;

int N_PARTICLES = 100000;
float DT = 0.01f;
int NSteps = 100;
float Lx = 1.0f;
float Ly = 1.0f;
float dx = 0.01f;
float dy = 0.01f;
float Q_OVER_M = 1.0f;

int threadsPerBlock = 256;
int blocksPerGrid = 256;