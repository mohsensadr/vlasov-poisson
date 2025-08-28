#pragma once

#include "particle_container.cuh"
#include "field_container.cuh"
#include "sorting.cuh"
#include "constants.hpp"
#include <cuda_runtime.h>

class DepositorBase {
public:
    virtual ~DepositorBase() = default;

    virtual void deposit(
        ParticleContainer& pc,
        FieldContainer& fc,
        Sorting& sorter
    ) = 0;

protected:
    void sync() { cudaDeviceSynchronize(); }
};
