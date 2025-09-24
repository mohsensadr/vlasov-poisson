#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#include "Constants/constants.hpp"
#include "Solvers/solver.cuh"
#include "Initializations/initialization.cuh"
#include "IOs/IO.h"
#include "Depositors/moments.cuh"
#include "Containers/particle_container.cuh"
#include "Containers/field_container.cuh"
#include "Distributions/pdfs.cuh"
#include "Sorters/sorting.cuh"
#include "VRs/MxE.cuh"

void run(const std::string& pdf_type, float* pdf_params) {
    cudaMemcpyToSymbol(kb, &kb_host, sizeof(float));
    cudaMemcpyToSymbol(m, &m_host, sizeof(float));

    // TODO: dx, dy, Lx, Ly are member variables of field container, remove them from here.
    dx = Lx/N_GRID_X;
    dy = Ly/N_GRID_Y;
    grid_size = N_GRID_X*N_GRID_Y;

    ParticleContainer pc(N_PARTICLES);
    FieldContainer fc(N_GRID_X, N_GRID_Y, Lx, Ly);
    Sorting sorter(pc, fc);

    // Create the appropriate PDF struct for device use
    PDF_position pdf_position;
    if (pdf_type == "gaussian" || pdf_type == "Gaussian") {
        pdf_position = make_gaussian_pdf(pdf_params[0], Lx, Ly);
    } else if (pdf_type == "cosine" || pdf_type == "Cosine") {
        pdf_position = make_cosine_pdf(pdf_params[0], pdf_params[1], Lx, Ly);
    } else if (pdf_type == "double_gaussian" || pdf_type == "DoubleGaussian") {
        pdf_position = make_double_gaussian_pdf(pdf_params[0], pdf_params[1], pdf_params[2], pdf_params[3], 
                                              pdf_params[4], pdf_params[5], pdf_params[6], pdf_params[7], Lx, Ly);
    } else {
        throw std::invalid_argument("Unknown PDF type: " + pdf_type);
    }

    // initialize particle velocity and position
    initialize_particles<<<blocksPerGrid, threadsPerBlock>>>(
        pc.d_x, pc.d_y, pc.d_vx, pc.d_vy, Lx, Ly, N_PARTICLES, pdf_position
    );
    cudaDeviceSynchronize();

    // compute moments, needed to find emperical density field
    if (depositionMode == DepositionMode::SORTING) {
      sorter.sort_particles_by_cell();
    }
    compute_moments(pc, fc, sorter);
    cudaDeviceSynchronize();

    // set particle weights given estimted and exact fields
    initialize_weights<<<blocksPerGrid, threadsPerBlock>>>(
        pc.d_x, pc.d_y, fc.d_N, pc.d_w, N_PARTICLES, N_GRID_X, N_GRID_Y, Lx, Ly, pdf_position
    );
    cudaDeviceSynchronize();

    // recompute moments given weights, mainly for VR estimate
    if (depositionMode == DepositionMode::SORTING) {
      sorter.sort_particles_by_cell();
    }
    compute_moments(pc, fc, sorter);
    cudaDeviceSynchronize();

    // write out initial fields
    post_proc(fc, 0);
    cudaDeviceSynchronize();

    size_t size = N_PARTICLES * sizeof(float);

    for (int step = 1; step < NSteps+1; ++step) {
        // compute Electric field
        solve_poisson_periodic(fc);
        cudaDeviceSynchronize();

        // update wold given w
        cudaMemcpy(pc.d_wold, pc.d_w, size, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        // map weights from global to local eq.
        pc.map_weights(fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR, N_GRID_X, N_GRID_Y, Lx, Ly, true);
        cudaDeviceSynchronize();

        // Push particles in the velocity space
        // Use either MC or VR density estimtes in the rhs of the Poisson to get E
        if (rhsMode == RhsMode::VR)
          pc.update_velocity(fc.d_ExVR, fc.d_EyVR, N_GRID_X, N_GRID_Y, Lx, Ly, DT, Q_OVER_M);
        else
          pc.update_velocity(fc.d_Ex, fc.d_Ey, N_GRID_X, N_GRID_Y, Lx, Ly, DT, Q_OVER_M);
        cudaDeviceSynchronize();

        // map weights from local to global eq.
        pc.map_weights(fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_TVR, N_GRID_X, N_GRID_Y, Lx, Ly, false);
        cudaDeviceSynchronize();

        // MxE to conserve equil. moments.
        if (vrMode == VRMode::MXE) {
          update_weights_dispatch(pc.d_vx, pc.d_vy, sorter.d_cell_offsets, pc.d_w, pc.d_wold, fc.d_NVR, fc.d_UxVR, fc.d_UyVR, fc.d_ExVR, fc.d_EyVR, grid_size, Nm);
          cudaDeviceSynchronize();
        }
        
        // push particles in the position space
        pc.update_position(Lx, Ly, DT);
        cudaDeviceSynchronize();

        // update moments
        if (depositionMode == DepositionMode::SORTING) {
          sorter.sort_particles_by_cell();
          cudaDeviceSynchronize();
        }

        compute_moments(pc, fc, sorter);
        cudaDeviceSynchronize();

        // print output
        if (step % 10 == 0) {
            post_proc(fc, step);
            cudaDeviceSynchronize();
        }
    }

    std::cout << "Done.\n";
}

