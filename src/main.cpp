#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>  // C++17
#include <cstdio>      // for snprintf
#include "vlasov_poisson_pi.cuh"

#define N_PARTICLES 100000
#define N_GRID_X 128
#define N_GRID_Y 128
#define DT       0.05f
#define Lx       1.0f
#define Ly       1.0f
#define Q_OVER_M 1.0f

int main() {
    run();
    return 0;
}
