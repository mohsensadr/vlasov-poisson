// particle_container.cu
#include "particle_container.cuh"
#include <cmath>

ParticleContainer::ParticleContainer(int n_particles_) : n_particles(n_particles_) {
    size_t bytes = n_particles * sizeof(float);
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_vx, bytes);
    cudaMalloc(&d_vy, bytes);
    cudaMalloc(&d_w, bytes);
    cudaMalloc(&d_wold, bytes);
}

ParticleContainer::~ParticleContainer() {
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_w);
    cudaFree(d_wold);
}

__device__ int periodic_index(int i, int N) {
    return (i + N) % N;
}

void ParticleContainer::update_velocity(float *Ex, float *Ey,
                                        int N_GRID_X, int N_GRID_Y,
                                        float Lx, float Ly,
                                        float DT, float Q_OVER_M) 
{
    update_velocity_2d<<<blocksPerGrid, threadsPerBlock>>>(
        d_x, d_y, d_vx, d_vy, Ex, Ey, n_particles,
        N_GRID_X, N_GRID_Y, Lx, Ly, DT, Q_OVER_M
    );
    cudaDeviceSynchronize();
}

void ParticleContainer::update_position(float Lx, float Ly, float DT) {
    update_position_2d<<<blocksPerGrid, threadsPerBlock>>>(
        d_x, d_y, d_vx, d_vy, n_particles, Lx, Ly, DT
    );
    cudaDeviceSynchronize();
}

void ParticleContainer::map_weights(float *NVR, float *UxVR, float *UyVR, float *TVR,
                                    int N_GRID_X, int N_GRID_Y, float Lx, float Ly, bool global_to_local)
{
    map_weights_2d<<<blocksPerGrid, threadsPerBlock>>>(
        d_x, d_y, d_vx, d_vy, d_w, NVR, UxVR, UyVR, TVR, n_particles,
        N_GRID_X, N_GRID_Y, Lx, Ly, global_to_local
    );
    cudaDeviceSynchronize();
}

__global__ void update_velocity_2d(float *x, float *y, float *vx, float *vy,
                                  float *Ex, float *Ey, int n_particles,
            int N_GRID_X, int N_GRID_Y,
            float Lx, float Ly,
            float DT,
            float Q_OVER_M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    float xi = x[i] / Lx * N_GRID_X;
    float yi = y[i] / Ly * N_GRID_Y;

    int ix = floorf(xi);
    int iy = floorf(yi);
    float dx = xi - ix;
    float dy = yi - iy;

    int ix0 = periodic_index(ix, N_GRID_X);
    int ix1 = periodic_index(ix + 1, N_GRID_X);
    int iy0 = periodic_index(iy, N_GRID_Y);
    int iy1 = periodic_index(iy + 1, N_GRID_Y);

    float w00 = (1 - dx) * (1 - dy);
    float w01 = (1 - dx) * dy;
    float w10 = dx * (1 - dy);
    float w11 = dx * dy;

    int i00 = ix0 + iy0 * N_GRID_X;
    int i01 = ix0 + iy1 * N_GRID_X;
    int i10 = ix1 + iy0 * N_GRID_X;
    int i11 = ix1 + iy1 * N_GRID_X;

    float Exi = w00 * Ex[i00] + w01 * Ex[i01] + w10 * Ex[i10] + w11 * Ex[i11];
    float Eyi = w00 * Ey[i00] + w01 * Ey[i01] + w10 * Ey[i10] + w11 * Ey[i11];

    vx[i] += - Q_OVER_M * Exi * DT;
    vy[i] += - Q_OVER_M * Eyi * DT;
}

__global__ void update_position_2d(float *x, float *y, float *vx, float *vy,
                                  int n_particles, float Lx, float Ly, float DT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    x[i] += vx[i] * DT;
    y[i] += vy[i] * DT;

    // periodic boundaries
    if (x[i] < 0) x[i] += Lx;
    if (x[i] >= Lx) x[i] -= Lx;
    if (y[i] < 0) y[i] += Ly;
    if (y[i] >= Ly) y[i] -= Ly;
}

__global__ void map_weights_2d(float *x, float *y, float *vx, float *vy, float *w,
    float *NVR, float *UxVR, float *UyVR, float *TVR, int n_particles,
    int N_GRID_X, int N_GRID_Y, float Lx, float Ly, bool global_to_local
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        float Navg = (1.0f*n_particles) / (1.0f*N_GRID_X*N_GRID_Y);
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        float energy = (vx[i]-UxVR[idx])*(vx[i]-UxVR[idx]);
        energy += (vy[i]-UyVR[idx])*(vy[i]-UyVR[idx]);
        float energy0 = vx[i]*vx[i] + vy[i]*vy[i];
        float kbT_m = TVR[idx]; //kb/m=1
        float kbT_m0 = 1.0f; // kb/m=1;
        if(global_to_local){
          w[i] = w[i] * (NVR[idx]/Navg) * (kbT_m0/kbT_m) * expf(-energy/kbT_m/2.0f+energy0/kbT_m0/2.0f);
        }
        else{
          w[i] = w[i] * (Navg/NVR[idx]) * (kbT_m/kbT_m0) * expf(energy/kbT_m/2.0f-energy0/kbT_m0/2.0f);
        }
    }
}