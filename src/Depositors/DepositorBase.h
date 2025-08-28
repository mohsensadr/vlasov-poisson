#pragma once

#include "Containers/particle_container.cuh"
#include "Containers/field_container.cuh"
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
