// field_container.cuh
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

class FieldContainer {
public:
    float *d_N = nullptr;
    float *d_Ux = nullptr;
    float *d_Uy = nullptr;
    float *d_T = nullptr;
    float *d_Ex = nullptr;
    float *d_Ey = nullptr;

    float *d_NVR = nullptr;
    float *d_UxVR = nullptr;
    float *d_UyVR = nullptr;
    float *d_TVR = nullptr;

    int nx, ny;
    size_t grid_size;

    FieldContainer(int N_GRID_X, int N_GRID_Y) : nx(N_GRID_X), ny(N_GRID_Y) {
        grid_size = nx * ny;
        size_t bytes = grid_size * sizeof(float);

        cudaMalloc(&d_N, bytes);
        cudaMalloc(&d_Ux, bytes);
        cudaMalloc(&d_Uy, bytes);
        cudaMalloc(&d_T, bytes);
        cudaMalloc(&d_Ex, bytes);
        cudaMalloc(&d_Ey, bytes);

        cudaMalloc(&d_NVR, bytes);
        cudaMalloc(&d_UxVR, bytes);
        cudaMalloc(&d_UyVR, bytes);
        cudaMalloc(&d_TVR, bytes);
    }

    ~FieldContainer() {
        cudaFree(d_N);
        cudaFree(d_Ux);
        cudaFree(d_Uy);
        cudaFree(d_T);
        cudaFree(d_Ex);
        cudaFree(d_Ey);

        cudaFree(d_NVR);
        cudaFree(d_UxVR);
        cudaFree(d_UyVR);
        cudaFree(d_TVR);
    }

    // Optional: zero out all field arrays
    void clear() {
        size_t bytes = grid_size * sizeof(float);
        cudaMemset(d_N, 0, bytes);
        cudaMemset(d_Ux, 0, bytes);
        cudaMemset(d_Uy, 0, bytes);
        cudaMemset(d_T, 0, bytes);
        cudaMemset(d_Ex, 0, bytes);
        cudaMemset(d_Ey, 0, bytes);

        cudaMemset(d_NVR, 0, bytes);
        cudaMemset(d_UxVR, 0, bytes);
        cudaMemset(d_UyVR, 0, bytes);
        cudaMemset(d_TVR, 0, bytes);
    }
};
