#pragma once
#include "DepositorBase.h"

class BruteDepositor : public DepositorBase {
public:
    void deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter) override;
};
