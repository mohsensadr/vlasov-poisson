#include <math.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "VRs/MxE.cuh"
#include "Constants/constants.hpp"

template<int Nm>
__device__ void Gauss_Jordan(float_type H[Nm][Nm], float_type g[Nm], float_type x[Nm]) {
    for (int i = 0; i < Nm; i++) {
        float_type diag = H[i][i];
        for (int j = i; j < Nm; j++) H[i][j] /= diag;
        g[i] /= diag;

        for (int k = 0; k < Nm; k++) {
            if (k == i) continue;
            float_type factor = H[k][i];
            for (int j = i; j < Nm; j++) H[k][j] -= factor * H[i][j];
            g[k] -= factor * g[i];
        }
    }

    for (int i = 0; i < Nm; i++) x[i] = g[i];
}

template<int Nm>
__device__ float_type mom(float_type u1, float_type u2, float_type U_1, float_type U_2, int n) {
    switch(n) {
        case 0: return u1 - U_1;
        case 1: return u2 - U_2;
        case 2: return (u1 - U_1) * (u1 - U_1) + (u2 - U_2) * (u2 - U_2);
    }
    return 0.0;
}

template<int Nm>
__global__ void update_weights(
    const float_type* __restrict__ vx,
    const float_type* __restrict__ vy,
    const int* __restrict__ d_cell_offsets,
    float_type* __restrict__ w,
    float_type* __restrict__ wold,
    float_type* __restrict__ NVR,
    float_type* __restrict__ UxVR,
    float_type* __restrict__ UyVR,
    float_type* __restrict__ ExVR,
    float_type* __restrict__ EyVR,
    int num_cells,
    int n_particles,
    float_type dt
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    float_type Navg = float_type(n_particles) / float_type(num_cells);
    float_type tol = Tolerance<float_type>::value();
    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1];
    int Npc = end - start;

    if (Npc < 100) return;

    float_type p[Nm] = {0.0};
    float_type pt[Nm] = {0.0};
    float_type p0[Nm] = {0.0};
    p0[2] = 2.0;

    float_type sumwold = 0.0;

    for (int i = start; i < end; i++) {
        sumwold += wold[i];
        for (int j = 0; j < Nm; j++) {
            p[j] += mom<Nm>(vx[i], vy[i], 0.0, 0.0, j);
            pt[j] += (1.0 - wold[i]) * mom<Nm>(vx[i], vy[i], 0.0, 0.0, j);
        }
    }

    for (int i = 0; i < Nm; i++) {
        p[i] /= Npc;
        pt[i] /= NVR[cell];
        //printf("p[%d]=%f | ", i, p[i]);
        //printf("inter1 pt[%d]=%f | ", i, pt[i]);
    }

    for (int i = 0; i < Nm; i++) {
        pt[i] += p0[i];
        //printf("inter2 pt[%d]=%f |", i, pt[i]);
    }

    // correct moments using Ex and Ey
    pt[0] -= dt * ExVR[cell];
    pt[1] -= dt * EyVR[cell];
    pt[2] -= dt * (UxVR[cell]*ExVR[cell] + UyVR[cell]*EyVR[cell]);

    //for (int i = 0; i < Nm; i++) {
    //    printf("inter3 pt[%d]=%f | ", i, pt[i]);
    //}
    //printf("\n\n dt * ExVR[%d]=%f | ", cell, dt * ExVR[cell]);
    //printf("dt * EyVR[%d]=%f | ", cell, dt * EyVR[cell]);
    //printf("\ndt * (UxVR[%d]*ExVR[.] + UyVR[.]*EyVR[.]) = %f", cell, dt * (UxVR[cell]*ExVR[cell] + UyVR[cell]*EyVR[cell]));

    // now compute target <w*R(v)> moments: 
    // <R(v)>VR = <R(v)>0 + <(1-w)*R(v)> 
    // = <R(v)>0 + <R(v)> - <w*R(v)> 
    // which implies: <w*R(v)> = <R(v)>0 + <R(v)> - <R(v)>VR 
    // here we reuse variable p to denote <w*R(v)> from this point on
    for (int i = 0; i < Nm; i++) {
        p[i] = p0[i] + p[i] - pt[i];
        //printf("\n\n final target moment p[%d]=%f |", i, p[i]);
    }

    for (int i = start; i < end; i++)
        wold[i] = w[i];

    bool convergence = false;
    int max_iter = 1000;
    int iter = 0;
    float_type g[Nm], H[Nm][Nm], xvec[Nm], lam[Nm] = {0.0};

    while (!convergence) {
        iter++;
        if (iter > max_iter) break;

        // Compute gradient
        float_type res = 0.0;
        for (int j = 0; j < Nm; j++) {
            g[j] = 0.0;
            for (int i = start; i < end; i++)
                g[j] += w[i] * mom<Nm>(vx[i], vy[i], UxVR[cell], UyVR[cell], j);
            g[j] = g[j]/Npc - p[j];
            res += fabsf(g[j]);
        }
        if (res < tol){
          convergence = true;
          break;
        }

        // Compute Hessian
        for (int i = 0; i < Nm; i++)
            for (int j = 0; j < Nm; j++)
                H[i][j] = 0.0;

        for (int k = 0; k < Nm; k++) {
            for (int j = k; j < Nm; j++) {
                float_type Ski = 0.0, Sji = 0.0, SkiSji = 0.0;
                for (int i = start; i < end; i++) {
                    float_type mk = mom<Nm>(vx[i], vy[i], UxVR[cell], UyVR[cell], k);
                    float_type mj = mom<Nm>(vx[i], vy[i], UxVR[cell], UyVR[cell], j);
                    Ski += mk * w[i];
                    Sji += mj * w[i];
                    SkiSji += mk * mj * w[i];
                }
                H[k][j] = SkiSji/Npc - Ski/Npc*p[j] - Sji/Npc*p[k] + p[j]*p[k];
            }
        }

        for (int k = 0; k < Nm; k++)
            for (int j = 0; j < k; j++)
                H[k][j] = H[j][k];

        // Solve for Newton step
        Gauss_Jordan<Nm>(H, g, xvec);

        // Update weights
        float_type sumW = 0.0;
        for (int j = 0; j < Nm; j++)
            lam[j] -= xvec[j];

        for (int i = start; i < end; i++) {
            float_type dummy = 0.0;
            for (int j = 0; j < Nm; j++)
                dummy += lam[j]*(mom<Nm>(vx[i], vy[i], UxVR[cell], UyVR[cell], j)-p[j]);
            float_type dummy2 = expf(-dummy);
            w[i] = wold[i] / dummy2;
            sumW += w[i];
        }

        for (int i = start; i < end; i++)
            w[i] *= sumwold/sumW;
    }
    if(!convergence){
      for (int i = start; i < end; i++){
        w[i] = wold[i];
      }
    }
    if(iter > 999)
      printf("MxE iter %d in cell %d\n", iter, cell);
}

void update_weights_dispatch(
    const float_type* vx,
    const float_type* vy,
    const int* d_cell_offsets,
    float_type* w,
    float_type* wold,
    float_type* NVR,
    float_type* UxVR,
    float_type* UyVR,
    float_type* ExVR,
    float_type* EyVR,
    int num_cells,
    int Nm
) {

    switch (Nm) {
        case 3:
            update_weights<3><<<blocksPerGrid, threadsPerBlock>>>(vx, vy, d_cell_offsets, w, wold, NVR, UxVR, UyVR, ExVR, EyVR, num_cells, N_PARTICLES, DT);
            break;
        // Add more cases as needed
        default:
            printf("Unsupported Nm: %d\n", Nm);
            break;
    }
}
