// vlasov_poisson.cuh

#pragma once

void run(int N_GRID_X, int N_GRID_Y,
            int N_PARTICLES,
            float DT,
            int NSteps,
            float Lx,
            float Ly,
            int threadsPerBlock
            );