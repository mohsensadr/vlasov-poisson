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
    float_type var;
    float_type Lx, Ly;

    __device__ __host__ PDF_Template(float_type variance, float_type length_x, float_type length_y)
        : var(variance), Lx(length_x), Ly(length_y) {}

    __device__ __host__ float_type normalizer() const {
        return 2.0 * M_PI * var;
    }

    __device__ __host__ float_type pmax() const {
        return 1.0;
    }

    __device__ __host__ float_type operator()(float_type x, float_type y) const {
        float_type dx = x - Lx/2.0;
        float_type dy = y - Ly/2.0;
        return exp(-(dx*dx + dy*dy)/(2.0*var));
    }
};

// Cosine PDF specialization
template<>
struct PDF_Template<PDFType::COSINE> {
    float_type A;
    float_type kx;
    float_type Lx, Ly;

    __device__ __host__ PDF_Template(float_type amplitude, float_type wavenumber, float_type length_x, float_type length_y)
        : A(amplitude), kx(wavenumber), Lx(length_x), Ly(length_y) {}

    __device__ __host__ float_type normalizer() const {
        return (A * sin(Lx * kx) + Lx * kx) / kx;
    }

    __device__ __host__ float_type pmax() const {
        return (1.0 + A) / normalizer() * (1.0 / Ly);
    }

    __device__ __host__ float_type operator()(float_type x, float_type y) const {
        return (1.0 + A * cos(kx * x)) / normalizer() * (1.0 / Ly);
    }
};

// Uniform PDF specialization
template<>
struct PDF_Template<PDFType::UNIFORM> {
    float_type Lx, Ly;

    __device__ __host__ PDF_Template(float_type length_x, float_type length_y)
        : Lx(length_x), Ly(length_y) {}

    __device__ __host__ float_type normalizer() const {
        return Lx * Ly;
    }

    __device__ __host__ float_type pmax() const {
        return 1.0 / (Lx * Ly);
    }

    __device__ __host__ float_type operator()(float_type x, float_type y) const {
        return 1.0 / (Lx * Ly);
    }
};

// Double Gaussian PDF specialization
template<>
struct PDF_Template<PDFType::DOUBLE_GAUSSIAN> {
    float_type var1, var2;
    float_type x1, y1, x2, y2;
    float_type weight1, weight2;
    float_type Lx, Ly;

    __device__ __host__ PDF_Template(float_type variance1, float_type variance2, 
                                   float_type center1_x, float_type center1_y,
                                   float_type center2_x, float_type center2_y,
                                   float_type w1, float_type w2,
                                   float_type length_x, float_type length_y)
        : var1(variance1), var2(variance2), 
          x1(center1_x), y1(center1_y), x2(center2_x), y2(center2_y),
          weight1(w1), weight2(w2), Lx(length_x), Ly(length_y) {}

    __device__ __host__ float_type normalizer() const {
        return 2.0 * M_PI * (weight1 * var1 + weight2 * var2);
    }

    __device__ __host__ float_type pmax() const {
        return 1.0;
    }

    __device__ __host__ float_type operator()(float_type x, float_type y) const {
        float_type g1 = weight1 * exp(-((x-x1)*(x-x1) + (y-y1)*(y-y1))/(2.0*var1));
        float_type g2 = weight2 * exp(-((x-x2)*(x-x2) + (y-y2)*(y-y2))/(2.0*var2));
        return g1 + g2;
    }
};

// Type alias for convenience
using PDF_Gaussian = PDF_Template<PDFType::GAUSSIAN>;
using PDF_Cosine = PDF_Template<PDFType::COSINE>;
using PDF_Uniform = PDF_Template<PDFType::UNIFORM>;
using PDF_DoubleGaussian = PDF_Template<PDFType::DOUBLE_GAUSSIAN>;

#endif // PDF_TEMPLATES_CUH 