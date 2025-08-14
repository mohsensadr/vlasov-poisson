#include "MxE.cuh"
#include <math.h>
#include <cuda_runtime.h>

template<int Nm>
__device__ void Gauss_Jordan(float H[Nm][Nm], float g[Nm], float x[Nm]) {
    for (int i = 0; i < Nm; i++) {
        float diag = H[i][i];
        for (int j = i; j < Nm; j++) H[i][j] /= diag;
        g[i] /= diag;

        for (int k = 0; k < Nm; k++) {
            if (k == i) continue;
            float factor = H[k][i];
            for (int j = i; j < Nm; j++) H[k][j] -= factor * H[i][j];
            g[k] -= factor * g[i];
        }
    }

    for (int i = 0; i < Nm; i++) x[i] = g[i];
}

template<int Nm>
__device__ float mom(float u1, float u2, float U_1, float U_2, int n) {
    switch(n) {
        case 0: return u1 - U_1;
        case 1: return u2 - U_2;
        case 2: return 0.5f * ((u1 - U_1) * (u1 - U_1) + (u2 - U_2) * (u2 - U_2));
    }
    return 0.0f;
}

template<int Nm>
__global__ void update_weights(
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const int* __restrict__ d_cell_offsets,
    float* __restrict__ w,
    float* __restrict__ wold,
    float* __restrict__ UxVR,
    float* __restrict__ UyVR,
    int num_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;

    float tol = 1e-5f;
    int start = d_cell_offsets[cell];
    int end   = d_cell_offsets[cell + 1];
    int Npc = end - start;

    float p[Nm] = {0.0f};
    float pt[Nm] = {0.0f};
    float p0[Nm] = {0.0f};
    p0[2] = 1.0f;

    float sumwold = 0.0f;

    for (int i = start; i < end; i++) {
        sumwold += wold[i];
        for (int j = 0; j < Nm; j++) {
            p[j] += mom<Nm>(vx[i], vy[i], UxVR[cell], UyVR[cell], j);
            pt[j] += (1.0f - wold[i]) * mom<Nm>(vx[i], vy[i], UxVR[cell], UyVR[cell], j);
        }
    }

    for (int i = 0; i < Nm; i++) {
        p[i] /= Npc;
        pt[i] /= Npc;
        pt[i] += p0[i];
        p[i] = p0[i] + p[i] - pt[i];
    }

    for (int i = start; i < end; i++)
        wold[i] = w[i];

    bool convergence = false;
    int max_iter = 1000;
    int iter = 0;
    float g[Nm], H[Nm][Nm], xvec[Nm], lam[Nm] = {0.0f};

    while (!convergence) {
        iter++;
        if (iter > max_iter) break;

        // Compute gradient
        float res = 0.0f;
        for (int j = 0; j < Nm; j++) {
            g[j] = 0.0f;
            for (int i = start; i < end; i++)
                g[j] += w[i] * mom<Nm>(vx[i], vy[i], UxVR[cell], UyVR[cell], j);
            g[j] = g[j]/Npc - p[j];
            res += fabsf(g[j]);
        }
        if (res < tol) convergence = true;

        // Compute Hessian
        for (int i = 0; i < Nm; i++)
            for (int j = 0; j < Nm; j++)
                H[i][j] = 0.0f;

        for (int k = 0; k < Nm; k++) {
            for (int j = k; j < Nm; j++) {
                float Ski = 0.0f, Sji = 0.0f, SkiSji = 0.0f;
                for (int i = start; i < end; i++) {
                    float mk = mom<Nm>(vx[i], vy[i], UxVR[cell], UyVR[cell], k);
                    float mj = mom<Nm>(vx[i], vy[i], UxVR[cell], UyVR[cell], j);
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
        float sumW = 0.0f;
        for (int j = 0; j < Nm; j++)
            lam[j] -= xvec[j];

        for (int i = start; i < end; i++) {
            float dummy = 0.0f;
            for (int j = 0; j < Nm; j++)
                dummy += lam[j]*(mom<Nm>(vx[i], vy[i], UxVR[cell], UyVR[cell], j)-p[j]);
            float dummy2 = expf(-dummy);
            w[i] = wold[i] / dummy2;
            sumW += w[i];
        }

        for (int i = start; i < end; i++)
            w[i] *= sumwold/sumW;
    }
}

// Explicit instantiation for Nm = 3
template __global__ void update_weights<3>(
    const float*, const float*, const int*, float*, float*, float*, float*, int
);
