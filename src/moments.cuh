// moments.cuh

#pragma once
#include "Containers/particle_container.cuh"
#include "Containers/field_container.cuh"
#include "sorting.cuh"

void compute_moments(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter);
