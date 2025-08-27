#pragma once

#include "ParticleContainer.h"
#include "FieldContainer.h"
#include "Sorting.h"
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

    template<typename Kernel, typename... Args>
    void launch(Kernel kernel, dim3 grid, dim3 block, Args... args) {
        kernel<<<grid, block>>>(args...);
        sync();
    }
};
