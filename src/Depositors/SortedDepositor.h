#pragma once
#include "DepositorBase.h"

class SortedDepositor : public DepositorBase {
public:
    void deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter) override;
};
