#ifndef PDF_TEMPLATES_CUH
#define PDF_TEMPLATES_CUH

#include <cuda_runtime.h>
#include <cmath>

/**
 * @brief Template-based PDF system that's fully CUDA compatible
 * 
 * This approach uses template specialization to create different PDF types
 * that can be selected at compile time or through runtime dispatch.
 */

// PDF type enum
enum class PDFType {
    GAUSSIAN = 0,
    COSINE = 1,
    UNIFORM = 2,
    DOUBLE_GAUSSIAN = 3
};

// Base template - will be specialized for each PDF type
template<PDFType Type>
struct PDF_Template {
    // This should never be instantiated directly
    static_assert(false, "PDF_Template must be specialized for specific PDF types");
};

// Gaussian PDF specialization
template<>
struct PDF_Template<PDFType::GAUSSIAN> {
    float var;
    float Lx, Ly;

    __device__ __host__ PDF_Template(float variance, float length_x, float length_y)
        : var(variance), Lx(length_x), Ly(length_y) {}

    __device__ __host__ float normalizer() const {
        return 2.0f * M_PI * var;
    }

    __device__ __host__ float pmax() const {
        return 1.0f;
    }

    __device__ __host__ float operator()(float x, float y) const {
        float dx = x - Lx/2.0f;
        float dy = y - Ly/2.0f;
        return expf(-(dx*dx + dy*dy)/(2.0f*var));
    }
};

// Cosine PDF specialization
template<>
struct PDF_Template<PDFType::COSINE> {
    float A;
    float kx;
    float Lx, Ly;

    __device__ __host__ PDF_Template(float amplitude, float wavenumber, float length_x, float length_y)
        : A(amplitude), kx(wavenumber), Lx(length_x), Ly(length_y) {}

    __device__ __host__ float normalizer() const {
        return (A * sinf(Lx * kx) + Lx * kx) / kx;
    }

    __device__ __host__ float pmax() const {
        return (1.0f + A) / normalizer() * (1.0f / Ly);
    }

    __device__ __host__ float operator()(float x, float y) const {
        return (1.0f + A * cosf(kx * x)) / normalizer() * (1.0f / Ly);
    }
};

// Uniform PDF specialization
template<>
struct PDF_Template<PDFType::UNIFORM> {
    float Lx, Ly;

    __device__ __host__ PDF_Template(float length_x, float length_y)
        : Lx(length_x), Ly(length_y) {}

    __device__ __host__ float normalizer() const {
        return Lx * Ly;
    }

    __device__ __host__ float pmax() const {
        return 1.0f / (Lx * Ly);
    }

    __device__ __host__ float operator()(float x, float y) const {
        return 1.0f / (Lx * Ly);
    }
};

// Double Gaussian PDF specialization
template<>
struct PDF_Template<PDFType::DOUBLE_GAUSSIAN> {
    float var1, var2;
    float x1, y1, x2, y2;
    float weight1, weight2;
    float Lx, Ly;

    __device__ __host__ PDF_Template(float variance1, float variance2, 
                                   float center1_x, float center1_y,
                                   float center2_x, float center2_y,
                                   float w1, float w2,
                                   float length_x, float length_y)
        : var1(variance1), var2(variance2), 
          x1(center1_x), y1(center1_y), x2(center2_x), y2(center2_y),
          weight1(w1), weight2(w2), Lx(length_x), Ly(length_y) {}

    __device__ __host__ float normalizer() const {
        return 2.0f * M_PI * (weight1 * var1 + weight2 * var2);
    }

    __device__ __host__ float pmax() const {
        return 1.0f;
    }

    __device__ __host__ float operator()(float x, float y) const {
        float g1 = weight1 * expf(-((x-x1)*(x-x1) + (y-y1)*(y-y1))/(2.0f*var1));
        float g2 = weight2 * expf(-((x-x2)*(x-x2) + (y-y2)*(y-y2))/(2.0f*var2));
        return g1 + g2;
    }
};

// Type alias for convenience
using PDF_Gaussian = PDF_Template<PDFType::GAUSSIAN>;
using PDF_Cosine = PDF_Template<PDFType::COSINE>;
using PDF_Uniform = PDF_Template<PDFType::UNIFORM>;
using PDF_DoubleGaussian = PDF_Template<PDFType::DOUBLE_GAUSSIAN>;

#endif // PDF_TEMPLATES_CUH 