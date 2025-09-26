// field_container.cuh
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include "Constants/constants.hpp"

#define TILE_X 16
#define TILE_Y 16

class FieldContainer {
public:
    float_type *d_N = nullptr;
    float_type *d_Ux = nullptr;
    float_type *d_Uy = nullptr;
    float_type *d_T = nullptr;
    float_type *d_phi = nullptr;
    float_type *d_Ex = nullptr;
    float_type *d_Ey = nullptr;

    float_type *d_NVR = nullptr;
    float_type *d_UxVR = nullptr;
    float_type *d_UyVR = nullptr;
    float_type *d_TVR = nullptr;
    float_type *d_phiVR = nullptr;
    float_type *d_ExVR = nullptr;
    float_type *d_EyVR = nullptr;

    float_type dx, dy;
    float_type xmin, ymin;
    int nx, ny;
    size_t grid_size;

    FieldContainer(int N_GRID_X, int N_GRID_Y, float_type Lx, float_type Ly) : nx(N_GRID_X), ny(N_GRID_Y) {
        grid_size = nx * ny;
        xmin = 0.0;
        ymin = 0.0;
        dx = Lx / nx;
        dy = Ly / ny;
        size_t bytes = grid_size * sizeof(float_type);

        cudaMalloc(&d_N, bytes);
        cudaMalloc(&d_Ux, bytes);
        cudaMalloc(&d_Uy, bytes);
        cudaMalloc(&d_T, bytes);
        cudaMalloc(&d_phi, bytes);
        cudaMalloc(&d_Ex, bytes);
        cudaMalloc(&d_Ey, bytes);

        cudaMalloc(&d_NVR, bytes);
        cudaMalloc(&d_UxVR, bytes);
        cudaMalloc(&d_UyVR, bytes);
        cudaMalloc(&d_TVR, bytes);
        cudaMalloc(&d_phiVR, bytes);
        cudaMalloc(&d_ExVR, bytes);
        cudaMalloc(&d_EyVR, bytes);
    }

    ~FieldContainer() {
        cudaFree(d_N);
        cudaFree(d_Ux);
        cudaFree(d_Uy);
        cudaFree(d_T);
        cudaFree(d_phi);
        cudaFree(d_Ex);
        cudaFree(d_Ey);

        cudaFree(d_NVR);
        cudaFree(d_UxVR);
        cudaFree(d_UyVR);
        cudaFree(d_TVR);
        cudaFree(d_phiVR);
        cudaFree(d_ExVR);
        cudaFree(d_EyVR);
    }

    // Optional: zero out all field arrays
    void setZero() {
        size_t bytes = grid_size * sizeof(float_type);
        cudaMemset(d_N, 0, bytes);
        cudaMemset(d_Ux, 0, bytes);
        cudaMemset(d_Uy, 0, bytes);
        cudaMemset(d_T, 0, bytes);

        cudaMemset(d_NVR, 0, bytes);
        cudaMemset(d_UxVR, 0, bytes);
        cudaMemset(d_UyVR, 0, bytes);
        cudaMemset(d_TVR, 0, bytes);
    }
};
