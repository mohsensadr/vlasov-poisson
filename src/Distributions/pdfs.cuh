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
    float_type params[8];  // Store all parameters in a flat array
    float_type Lx, Ly;
    
    __host__ __device__
    PDF_position() : type(-1), Lx(1.0), Ly(1.0) {
        for (int i = 0; i < 8; ++i) params[i] = 0.0;
    }

    // Constructor for Gaussian
    __device__ __host__ PDF_position(float_type variance, float_type length_x, float_type length_y) 
        : type(0), Lx(length_x), Ly(length_y) {
        params[0] = variance;
    }
    
    // Constructor for Cosine
    __device__ __host__ PDF_position(float_type amplitude, float_type wavenumber, float_type length_x, float_type length_y, int cosine_type) 
        : type(1), Lx(length_x), Ly(length_y) {
        params[0] = amplitude;
        params[1] = wavenumber;
        params[2]  = (params[0] * sinf(Lx * wavenumber) + Lx * wavenumber) / wavenumber ;
        params[2] *= (params[0] * sinf(Ly * wavenumber) + Ly * wavenumber) / wavenumber ;
    }
    
    // Constructor for Double Gaussian
    __device__ __host__ PDF_position(float_type var1, float_type var2, float_type x1, float_type y1, 
                                   float_type x2, float_type y2, float_type w1, float_type w2,
                                   float_type length_x, float_type length_y, int double_gaussian_type) 
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

    __device__ __host__ float_type normalizer() const {
        switch(type) {
            case 0: // gaussian
                return 1.0;
            case 1: // cosine
                return (params[0] * sin(Lx * params[1]) + Lx * params[1]) / params[1] * (params[0] * sin(Ly * params[1]) + Ly * params[1]) / params[1];
            case 2: // double_gaussian
                return 2.0 * PI_F * (params[6] * params[0] + params[7] * params[1]);
            default:
                return 1.0;
        }
    }

    __device__ __host__ float_type pmax() const {
        switch(type) {
            case 0: // gaussian
                return 1.0 / (2.0 * PI_F * params[0]);
            case 1: // cosine
                return  (1.0 + params[0]) / params[2] * (1.0 + params[0]) / params[2];
            case 2: // double_gaussian
                return 1.0;
            default:
                return 1.0;
        }
    }

    __device__ __host__ float_type operator()(float_type x, float_type y) const {
        switch(type) {
            case 0: { // gaussian
                float_type dx = x - Lx/2.0;
                float_type dy = y - Ly/2.0;
                return exp(-(dx*dx + dy*dy)/(2.0*params[0])) / (2.0 * PI_F * params[0]);
            }
            case 1: // cosine
                return (1.0 + params[0] * cos(params[1] * x)) / params[2] * (1.0 + params[0] * cos(params[1] * y)) / params[2];
            case 2: { // double_gaussian
                float_type g1 = params[6] * exp(-((x-params[2])*(x-params[2]) + (y-params[3])*(y-params[3]))/(2.0*params[0]));
                float_type g2 = params[7] * exp(-((x-params[4])*(x-params[4]) + (y-params[5])*(y-params[5]))/(2.0*params[1]));
                return g1 + g2;
            }
            default:
                return 1.0;
        }
    }
};

// Helper functions to create PDF instances
__device__ __host__ inline PDF_position make_gaussian_pdf(float_type variance, float_type Lx, float_type Ly) {
    return PDF_position(variance, Lx, Ly);
}

__device__ __host__ inline PDF_position make_cosine_pdf(float_type amplitude, float_type wavenumber, float_type Lx, float_type Ly) {
    return PDF_position(amplitude, wavenumber, Lx, Ly, 1);
}

__device__ __host__ inline PDF_position make_double_gaussian_pdf(float_type var1, float_type var2, 
                                                               float_type x1, float_type y1, float_type x2, float_type y2,
                                                               float_type w1, float_type w2, float_type Lx, float_type Ly) {
    return PDF_position(var1, var2, x1, y1, x2, y2, w1, w2, Lx, Ly, 2);
}

#endif // PDF_CUH 