// vlasov_poisson_pi.cuh

#pragma once

// Declare your kernel
__global__ void deposit_charge_2d(float* d_x, float* d_y, float* d_rho, int N);

void run();