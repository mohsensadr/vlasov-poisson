

__global__ void deposit_density_2d(float *x, float *y, float *N, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&N[idx], 1.0f);
    }
}

__global__ void deposit_velocity_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&Ux[idx], vx[i]/N[idx]);
        atomicAdd(&Uy[idx], vy[i]/N[idx]);
    }
}

__global__ void deposit_temperature_2d(float *x, float *y, float *N, float *vx, float *vy, float *Ux, float *Uy, float *T, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        float energy = (vx[i]-Ux[idx])*(vx[i]-Ux[idx]);
        energy += (vy[i]-Uy[idx])*(vy[i]-Uy[idx]);
        atomicAdd(&T[idx], energy/N[idx]); 
    }
}

void compute_moments(float *d_x, float *d_y, float *d_vx, float *d_vy, 
    float *d_N, float *d_Ux, float *d_Uy, float *d_T,
    int n_particles, int N_GRID_X, int N_GRID_Y, float Lx, float Ly,
    int blocksPerGrid, int threadsPerBlock){

    // compute number of particles in each cell         
    deposit_density_2d<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_N, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    // compute average velocity in each cell
    deposit_velocity_2d<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_N, d_vx, d_vy, d_Ux, d_Uy, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);

    // compute temperature in each cell
    deposit_temperature_2d<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_N, d_vx, d_vy, d_Ux, d_Uy, d_T, n_particles, N_GRID_X, N_GRID_Y, Lx, Ly);
}

