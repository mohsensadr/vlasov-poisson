// vlasov_poisson_pi.cuh

#pragma once

void run(int N_GRID_X, int N_GRID_Y,
            int N_PARTICLES,
            float DT,
            float Lx,
            float Ly,
            float Q_OVER_M
            );