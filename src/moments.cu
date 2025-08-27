#include "constants.hpp"
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "moments.cuh"
#include "Depositors/BruteDepositor.h"
#include "Depositors/TiledDepositor.h"
#include "Depositors/SortedDepositor.h"

void compute_moments(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter) {
    std::unique_ptr<MomentDepositor> depositor;

    switch (depositionMode) {
        case DepositionMode::BRUTE:   depositor = std::make_unique<BruteDepositor>(); break;
        case DepositionMode::TILING:  depositor = std::make_unique<TiledDepositor>(); break;
        case DepositionMode::SORTING: depositor = std::make_unique<SortingDepositor>(sorter); break;
    }

    fc.setZero();
    cudaMemcpyToSymbol(kb, &kb_host, sizeof(float));
    cudaMemcpyToSymbol(m, &m_host, sizeof(float));

    depositor->deposit(pc, fc, sorter);
}