%%writefile vlasov_poisson_pic.cu
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define N_PARTICLES 100000
#define N_GRID_X 128
#define N_GRID_Y 128
#define DT       0.05f
#define Lx       1.0f
#define Ly       1.0f
#define Q_OVER_M 1.0f

__device__ int periodic_index(int i, int N) {
    return (i + N) % N;
}

__global__ void deposit_charge_2d(float *x, float *y, float *rho, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        int ix = int(x[i] / Lx * N_GRID_X) % N_GRID_X;
        int iy = int(y[i] / Ly * N_GRID_Y) % N_GRID_Y;
        int idx = ix + iy * N_GRID_X;
        atomicAdd(&rho[idx], 1.0f);  // ensure atomic
    }
}

/*
__global__ void deposit_charge_2d(float *x, float *y, float *rho, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    float xi = x[i] / Lx * N_GRID_X;
    float yi = y[i] / Ly * N_GRID_Y;

    int ix = floorf(xi);
    int iy = floorf(yi);
    float dx = xi - ix;
    float dy = yi - iy;

    float w00 = (1 - dx) * (1 - dy);
    float w01 = (1 - dx) * dy;
    float w10 = dx * (1 - dy);
    float w11 = dx * dy;

    int i00 = periodic_index(ix, N_GRID_X) + periodic_index(iy, N_GRID_Y) * N_GRID_X;
    int i01 = periodic_index(ix, N_GRID_X) + periodic_index(iy + 1, N_GRID_Y) * N_GRID_X;
    int i10 = periodic_index(ix + 1, N_GRID_X) + periodic_index(iy, N_GRID_Y) * N_GRID_X;
    int i11 = periodic_index(ix + 1, N_GRID_X) + periodic_index(iy + 1, N_GRID_Y) * N_GRID_X;

    atomicAdd(&rho[i00], w00);
    atomicAdd(&rho[i01], w01);
    atomicAdd(&rho[i10], w10);
    atomicAdd(&rho[i11], w11);
}
*/

__global__ void push_particles_2d(float *x, float *y, float *vx, float *vy,
                                  float *Ex, float *Ey, int n_particles) {
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

    vx[i] += Q_OVER_M * Exi * DT;
    vy[i] += Q_OVER_M * Eyi * DT;

    x[i] += vx[i] * DT;
    y[i] += vy[i] * DT;

    // periodic boundaries
    if (x[i] < 0) x[i] += Lx;
    if (x[i] >= Lx) x[i] -= Lx;
    if (y[i] < 0) y[i] += Ly;
    if (y[i] >= Ly) y[i] -= Ly;
}

// Placeholder Poisson solver: sets E = 0 for demo (replace with real solver)
void solve_poisson(float *rho, float *Ex, float *Ey) {
    int size = N_GRID_X * N_GRID_Y;
    for (int i = 0; i < size; ++i) {
        Ex[i] = 0.0f;
        Ey[i] = 0.0f;
    }
}

void write_density_to_csv(const char *filename, float *rho) {
    std::ofstream out(filename);
    for (int j = 0; j < N_GRID_Y; ++j) {
        for (int i = 0; i < N_GRID_X; ++i) {
            out << rho[i + j * N_GRID_X];
            if (i < N_GRID_X - 1) out << ",";
        }
        out << "\n";
    }
    out.close();
}

int main() {
    float *x = new float[N_PARTICLES];
    float *y = new float[N_PARTICLES];
    float *vx = new float[N_PARTICLES];
    float *vy = new float[N_PARTICLES];

    for (int i = 0; i < N_PARTICLES; ++i) {
        float rx = static_cast<float>(rand()) / RAND_MAX;
        float ry = static_cast<float>(rand()) / RAND_MAX;

        x[i] = Lx * rx;
        y[i] = Ly * ry;

        // Maxwellian in vx, vy (Box-Muller)
        float u1 = static_cast<float>(rand()) / RAND_MAX;
        float u2 = static_cast<float>(rand()) / RAND_MAX;
        float u3 = static_cast<float>(rand()) / RAND_MAX;
        float u4 = static_cast<float>(rand()) / RAND_MAX;

        vx[i] = sqrtf(-2 * log(u1)) * cosf(2 * M_PI * u2);
        vy[i] = sqrtf(-2 * log(u3)) * cosf(2 * M_PI * u4);

        // Landau perturbation in density
        float alpha = 0.01f;
        vx[i] += alpha * cosf(2 * M_PI * x[i] / Lx);
        vy[i] += alpha * cosf(2 * M_PI * y[i] / Ly);
    }


    float *d_x, *d_y, *d_vx, *d_vy;
    float *d_rho, *d_Ex, *d_Ey;

    cudaMalloc(&d_x, sizeof(float) * N_PARTICLES);
    cudaMalloc(&d_y, sizeof(float) * N_PARTICLES);
    cudaMalloc(&d_vx, sizeof(float) * N_PARTICLES);
    cudaMalloc(&d_vy, sizeof(float) * N_PARTICLES);

    int grid_size = N_GRID_X * N_GRID_Y;
    cudaMalloc(&d_rho, sizeof(float) * grid_size);
    cudaMalloc(&d_Ex, sizeof(float) * grid_size);
    cudaMalloc(&d_Ey, sizeof(float) * grid_size);

    cudaMemcpy(d_x, x, sizeof(float) * N_PARTICLES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * N_PARTICLES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, vx, sizeof(float) * N_PARTICLES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, vy, sizeof(float) * N_PARTICLES, cudaMemcpyHostToDevice);

    for (int step = 0; step < 200; ++step) {
        cudaMemset(d_rho, 0, sizeof(float) * grid_size);

        deposit_charge_2d<<<(N_PARTICLES + 255) / 256, 256>>>(d_x, d_y, d_rho, N_PARTICLES);
        cudaDeviceSynchronize();

        float *rho_host = new float[grid_size];
        float *Ex_host = new float[grid_size];
        float *Ey_host = new float[grid_size];

        cudaMemcpy(rho_host, d_rho, sizeof(float) * grid_size, cudaMemcpyDeviceToHost);

        solve_poisson(rho_host, Ex_host, Ey_host); // (still a placeholder)

        cudaMemcpy(d_Ex, Ex_host, sizeof(float) * grid_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ey, Ey_host, sizeof(float) * grid_size, cudaMemcpyHostToDevice);

        push_particles_2d<<<(N_PARTICLES + 255) / 256, 256>>>(d_x, d_y, d_vx, d_vy, d_Ex, d_Ey, N_PARTICLES);
        cudaDeviceSynchronize();

        // Save rho every 10 steps
        if (step % 10 == 0) {
            char filename[64];
            snprintf(filename, sizeof(filename), "rho_step_%03d.csv", step);
            write_density_to_csv(filename, rho_host);
            std::cout << "Saved: " << filename << "\n";
        }

        delete[] rho_host;
        delete[] Ex_host;
        delete[] Ey_host;
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_vx); cudaFree(d_vy);
    cudaFree(d_rho); cudaFree(d_Ex); cudaFree(d_Ey);
    delete[] x; delete[] y; delete[] vx; delete[] vy;

    std::cout << "Done. Charge density saved to rho.csv\n";
    return 0;
}

