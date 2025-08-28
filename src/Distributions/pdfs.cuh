#ifndef PDF_CUH
#define PDF_CUH

#include <cuda_runtime.h>
#include <cmath>

/**
 * @brief Simple CUDA-compatible PDF structure
 * 
 * This approach uses a single struct with a type field and parameters.
 * It's more straightforward and guaranteed to work with CUDA.
 */

struct PDF_position {
    int type;  // 0=gaussian, 1=cosine, 2=double_gaussian
    
    // Parameters for all PDF types
    float params[8];  // Store all parameters in a flat array
    float Lx, Ly;
    
    __host__ __device__
    PDF_position() : type(-1), Lx(1.0f), Ly(1.0f) {
        for (int i = 0; i < 8; ++i) params[i] = 0.0f;
    }

    // Constructor for Gaussian
    __device__ __host__ PDF_position(float variance, float length_x, float length_y) 
        : type(0), Lx(length_x), Ly(length_y) {
        params[0] = variance;
    }
    
    // Constructor for Cosine
    __device__ __host__ PDF_position(float amplitude, float wavenumber, float length_x, float length_y, int cosine_type) 
        : type(1), Lx(length_x), Ly(length_y) {
        params[0] = amplitude;
        params[1] = wavenumber;
        params[2] = (params[0] * sinf(Lx * wavenumber) + Lx * wavenumber) / wavenumber ;
    }
    
    // Constructor for Double Gaussian
    __device__ __host__ PDF_position(float var1, float var2, float x1, float y1, 
                                   float x2, float y2, float w1, float w2,
                                   float length_x, float length_y, int double_gaussian_type) 
        : type(2), Lx(length_x), Ly(length_y) {
        params[0] = var1;
        params[1] = var2;
        params[2] = x1;
        params[3] = y1;
        params[4] = x2;
        params[5] = y2;
        params[6] = w1;
        params[7] = w2;
    }

    __device__ __host__ float normalizer() const {
        switch(type) {
            case 0: // gaussian
                return 1.0;
            case 1: // cosine
                return (params[0] * sinf(Lx * params[1]) + Lx * params[1]) / params[1] * 1.0f / Ly;
            case 2: // double_gaussian
                return 2.0f * PI_F * (params[6] * params[0] + params[7] * params[1]);
            default:
                return 1.0f;
        }
    }

    __device__ __host__ float pmax() const {
        switch(type) {
            case 0: // gaussian
                return 1.0f / (2.0f * PI_F * params[0]);
            case 1: // cosine
                return  (1.0f + params[0]) / params[2] * (1.0f / Ly);
            case 2: // double_gaussian
                return 1.0f;
            default:
                return 1.0f;
        }
    }

    __device__ __host__ float operator()(float x, float y) const {
        switch(type) {
            case 0: { // gaussian
                float dx = x - Lx/2.0f;
                float dy = y - Ly/2.0f;
                return expf(-(dx*dx + dy*dy)/(2.0f*params[0])) / (2.0f * PI_F * params[0] );
            }
            case 1: // cosine
                return (1.0f + params[0] * cosf(params[1] * x)) / params[2] * (1.0f / Ly);
            case 2: { // double_gaussian
                float g1 = params[6] * expf(-((x-params[2])*(x-params[2]) + (y-params[3])*(y-params[3]))/(2.0f*params[0]));
                float g2 = params[7] * expf(-((x-params[4])*(x-params[4]) + (y-params[5])*(y-params[5]))/(2.0f*params[1]));
                return g1 + g2;
            }
            default:
                return 1.0f;
        }
    }
};

// Helper functions to create PDF instances
__device__ __host__ inline PDF_position make_gaussian_pdf(float variance, float Lx, float Ly) {
    return PDF_position(variance, Lx, Ly);
}

__device__ __host__ inline PDF_position make_cosine_pdf(float amplitude, float wavenumber, float Lx, float Ly) {
    return PDF_position(amplitude, wavenumber, Lx, Ly, 1);
}

__device__ __host__ inline PDF_position make_double_gaussian_pdf(float var1, float var2, 
                                                               float x1, float y1, float x2, float y2,
                                                               float w1, float w2, float Lx, float Ly) {
    return PDF_position(var1, var2, x1, y1, x2, y2, w1, w2, Lx, Ly, 2);
}

#endif // PDF_CUH 