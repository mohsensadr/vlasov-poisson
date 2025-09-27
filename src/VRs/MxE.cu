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

// --- Device Gauss-Jordan with partial pivoting ---
// Returns true on success, false if matrix is singular or non-finite values appear.
// Overwrites H and g; x out receives the solution if returned true.
template<int Nm>
__device__ bool Gauss_Jordan_pivot(float_type H[Nm][Nm], float_type g[Nm], float_type x[Nm]) {
    const float_type TINY = (float_type)1e-12; // tune as needed

    for (int i = 0; i < Nm; ++i) {
        // find pivot (largest magnitude in column i among rows i..Nm-1)
        int pivot = i;
        float_type maxabs = fabsf(H[i][i]);
        for (int r = i+1; r < Nm; ++r) {
            float_type av = fabsf(H[r][i]);
            if (av > maxabs) { maxabs = av; pivot = r; }
        }

        if (maxabs < TINY || !isfinite(maxabs)) {
            // near singular pivot
            return false;
        }

        // swap rows if necessary
        if (pivot != i) {
            for (int c = 0; c < Nm; ++c) {
                float_type tmp = H[i][c];
                H[i][c] = H[pivot][c];
                H[pivot][c] = tmp;
            }
            float_type tmpg = g[i];
            g[i] = g[pivot];
            g[pivot] = tmpg;
        }

        // normalize pivot row (use reciprocal)
        float_type diag = H[i][i];
        if (!isfinite(diag) || fabsf(diag) < TINY) return false;
        float_type invd = (float_type)1.0 / diag;
        for (int c = i; c < Nm; ++c) H[i][c] *= invd;
        g[i] *= invd;

        // eliminate column i from other rows
        for (int r = 0; r < Nm; ++r) {
            if (r == i) continue;
            float_type factor = H[r][i];
            // subtract factor * pivot-row (only columns c >= i are used)
            for (int c = i; c < Nm; ++c) H[r][c] -= factor * H[i][c];
            g[r] -= factor * g[i];
        }

        // optional: quick finite check
        for (int rr = 0; rr < Nm; ++rr) {
            for (int cc = 0; cc < Nm; ++cc) {
                if (!isfinite(H[rr][cc])) return false;
            }
            if (!isfinite(g[rr])) return false;
        }
    }

    // at this point g contains the solution
    for (int i = 0; i < Nm; ++i) x[i] = g[i];
    return true;
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
    const float_type* __restrict__ vx_old,
    const float_type* __restrict__ vy_old,
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
    float_type Ux = UxVR[cell];
    float_type Uy = UyVR[cell];
    float_type tol = Tolerance<float_type>::value();
    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1];
    int Npc = end - start;

    if (Npc < 100) return;

    float_type p[Nm] = {0.0};
    float_type pt[Nm] = {0.0};
    float_type p0[Nm] = {0.0};
    float_type pw[Nm] = {0.0};
    p0[2] = 2.0*Navg; // energy is assumed 1.0 for the control variate here.

    float_type sumwold = 0.0;

    for (int i = start; i < end; i++) {
        sumwold += wold[i];
        for (int j = 0; j < Nm; j++) {
            p[j] += mom<Nm>(vx[i], vy[i], Ux, Uy, j);
            pt[j] += (1.0 - wold[i]) * mom<Nm>(vx_old[i], vy_old[i], Ux, Uy, j);
            pw[j] += w[i] * mom<Nm>(vx[i], vy[i], Ux, Uy, j);
        }
    }

    /*
    printf("\n <R>: ");
    for (int i = 0; i < Nm; i++) {
        printf("[%d]=%f | ", i, p[i]);
    }

    printf("\n <(1-w)R>:");
    for (int i = 0; i < Nm; i++) {
        printf("[%d]=%f | ", i, pt[i]);
    }*/

    for (int i = 0; i < Nm; i++) {
        pt[i] += p0[i];
    }
    /*
    printf("\n <R>0 + <(1-w)R>:");
    for (int i = 0; i < Nm; i++) {
        printf(" [%d]=%f |", i, pt[i]);
    }*/

    // correct moments using Ex and Ey
    pt[0] = dt * NVR[cell]*ExVR[cell];
    pt[1] = dt * NVR[cell]*EyVR[cell];
    //pt[2] -= dt * NVR[cell]*(UxVR[cell]*ExVR[cell] + UyVR[cell]*EyVR[cell]);

    /*
    printf("\n <R>_{t+dt}:");
    for (int i = 0; i < Nm; i++) {
        printf("[%d]=%f | ", i, pt[i]);
    }*/

    // now compute target <w*R(v)> moments: 
    // <R(v)>VR = <R(v)>0 + <(1-w)*R(v)> 
    // = <R(v)>0 + <R(v)> - <w*R(v)> 
    // which implies: <w*R(v)> = <R(v)>0 + <R(v)> - <R(v)>VR 
    // here we reuse variable p to denote <w*R(v)> from this point on
    for (int i = 0; i < Nm; i++) {
        p[i] = p0[i] + p[i] - pt[i];
    }

    /*
    printf("\n <WR>_{old}:");
    for (int i = 0; i < Nm; i++) {
        printf("[%d]=%f | ", i, pw[i]);
    }

    printf("\n <WR>_{t+dt}:");
    for (int i = 0; i < Nm; i++) {
        printf("[%d]=%f | ", i, p[i]);
    }*/


    for (int i = 0; i < Nm; i++) {
        p[i] = p[i]/Npc;
    }

    /*
    printf("\n <WR>_{t+dt}/Npc:");
    for (int i = 0; i < Nm; i++) {
        printf("[%d]=%f | ", i, p[i]);
    }*/

    for (int i = start; i < end; i++)
        wold[i] = w[i];

    bool convergence = false;
    int max_iter = 1000;
    int iter = 0;
    float_type g[Nm], H[Nm][Nm], xvec[Nm], lam[Nm] = {0.0};

    float_type res;
    while (!convergence) {
        iter++;
        if (iter > max_iter) break;

        // Compute gradient
        res = 0.0;
        for (int j = 0; j < Nm; j++) {
            g[j] = 0.0;
            for (int i = start; i < end; i++)
                g[j] += w[i] * mom<Nm>(vx[i], vy[i], Ux, Uy, j);
            g[j] = g[j]/Npc - p[j];
            res += fabs(g[j]);
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
                    float_type mk = mom<Nm>(vx[i], vy[i], Ux, Uy, k);
                    float_type mj = mom<Nm>(vx[i], vy[i], Ux, Uy, j);
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
        //Gauss_Jordan<Nm>(H, g, xvec);
        bool ok = Gauss_Jordan_pivot<Nm>(H, g, xvec);
        if (!ok) {
            printf("Solver failed! Don't update weights\n");
            // fallback: do not overwrite weights if solver failed
            for (int i = start; i < end; ++i) w[i] = wold[i];
            // optionally break out of outer loop or mark no convergence
            convergence = false;
            printf("res: %e\n", res);
            break; // or set iter=max_iter to exit
        }

        // Update weights
        float_type sumW = 0.0;
        for (int j = 0; j < Nm; j++)
            lam[j] -= xvec[j];

        for (int i = start; i < end; i++) {
            float_type dummy = 0.0;
            for (int j = 0; j < Nm; j++)
                dummy += lam[j]*(mom<Nm>(vx[i], vy[i], Ux, Uy, j)-p[j]);
            float_type dummy2 = exp(-dummy);
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
    if(iter > 999){
      printf("iter: %d res: %e\n", iter, res);
      printf("MxE iter %d in cell %d\n", iter, cell);
    }
}

void update_weights_dispatch(
    const float_type* vx,
    const float_type* vy,
    const float_type* vx_old,
    const float_type* vy_old,
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
            update_weights<3><<<blocksPerGrid, threadsPerBlock>>>(vx, vy, vx_old, vy_old, d_cell_offsets, w, wold, NVR, UxVR, UyVR, ExVR, EyVR, num_cells, N_PARTICLES, DT);
            break;
        // Add more cases as needed
        default:
            printf("Unsupported Nm: %d\n", Nm);
            break;
    }
}
