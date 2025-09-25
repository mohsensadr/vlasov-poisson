#pragma once
#include "DepositorBase.h"

class BruteDepositor : public DepositorBase {
public:
    void deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter) override;
};

__global__ void deposit_density_2d(float_type *x, float_type *y, float_type *N, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly
  );

inline void deposit_velocity_2d(float_type *x, float_type *y, float_type *N, float_type *vx, float_type *vy, float_type *Ux, float_type  *Uy, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly
  );

inline void deposit_temperature_2d(float_type *x, float_type *y, float_type *N, float_type *vx, float_type *vy, float_type *Ux, float_type *Uy, float_type *T, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly
  );

inline void deposit_density_2d_VR(float_type *x, float_type *y, float_type *w, float_type *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly
  );

inline void deposit_velocity_2d_VR(float_type *x, float_type *y, float_type *vx, float_type *vy, float_type *w,
            float_type *UxVR, float_type *UyVR, float_type *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly
    );

inline void deposit_temperature_2d_VR(float_type *x, float_type *y, float_type *vx, float_type *vy, float_type *w, float_type *N, float_type *NVR, float_type *UxVR, float_type *UyVR, float_type *TVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float_type Lx, float_type Ly
  );