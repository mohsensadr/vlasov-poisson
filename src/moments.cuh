// moments.cuh

#pragma once
#include "particle_container.cuh"
#include "field_container.cuh"
#include "sorting.cuh"

void compute_moments(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter);