#pragma once
#include "DepositorBase.h"

class TiledDepositor : public DepositorBase {
public:
    void deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter) override;
};
