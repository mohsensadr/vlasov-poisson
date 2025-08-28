#pragma once
#include "DepositorBase.h"

class BruteDepositor : public DepositorBase {
public:
    void deposit(ParticleContainer& pc, FieldContainer& fc, Sorting& sorter) override;
};

__global__ void deposit_density_2d(float *x, float *y, float *N, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  );

//__global__ void deposit_velocity_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, int n_particles,
//            int N_GRID_X, int N_GRID_Y,
//            float Lx, float Ly
//  );

inline void deposit_velocity_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  );

//__global__ void deposit_temperature_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, float *T, int n_particles,
//            int N_GRID_X, int N_GRID_Y,
//            float Lx, float Ly
//  );

inline void deposit_temperature_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, float *T, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  );

//__global__ void deposit_density_2d_VR(float *x, float *y, float *w, float *N, float *NVR, int n_particles,
//            int N_GRID_X, int N_GRID_Y,
//            float Lx, float Ly
//  );

inline void deposit_density_2d_VR(float *x, float *y, float *w, float *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  );

//__global__ void deposit_velocity_2d_VR(float *x, float *y, float *vx, float*vy, float *w,
//            float *UxVR, float *UyVR, float *NVR, int n_particles,
//            int N_GRID_X, int N_GRID_Y,
//            float Lx, float Ly
//  );

inline void deposit_velocity_2d_VR(float *x, float *y, float *vx, float*vy, float *w,
            float *UxVR, float *UyVR, float *NVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    );

//__global__ void deposit_temperature_2d_VR(float *x, float *y, float *vx, float *vy, float *w, float *N, float *NVR, float *UxVR, float *UyVR, float *TVR, int n_particles,
//            int N_GRID_X, int N_GRID_Y,
//            float Lx, float Ly
//  );

inline void deposit_temperature_2d_VR(float *x, float *y, float *vx, float *vy, float *w, float *N, float *NVR, float *UxVR, float *UyVR, float *TVR, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
  );