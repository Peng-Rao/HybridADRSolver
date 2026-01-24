/**
 * @file convergence_study.cpp
 * @brief Drivers for calculating convergence rates of Matrix-Based and
 * Matrix-Free solvers.
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "core/problem_definition.h"
#include "core/types.h"
#include "matrix_based/matrix_based_solver.h"
#include "matrix_free/matrix_free_solver.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

using namespace dealii;
using namespace HybridADRSolver;

// Helper to calculate grid size h for a Unit Hypercube
double compute_h(int refinement_level) {
    // For a unit hypercube, h = 1.0 / 2^refinement_level
    return 1.0 / std::pow(2.0, refinement_level);
}

int main(int argc, char* argv[]) {
    try {
        Utilities::MPI::MPI_InitFinalize mpi(argc, argv,
                                             numbers::invalid_unsigned_int);

        // Only write to CSV from the root process
        const bool is_root =
            (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

        // Setup Output File
        std::ofstream csv_file;
        if (is_root) {
            csv_file.open("convergence_results.csv");
            // Header
            csv_file << "refinement,h,time_mb,error_mb,time_mf,"
                        "error_mf\n";
            std::cout << "Running Convergence Study..." << std::endl;
        }

        // --- Study Parameters ---
        const int start_ref = 3;
        const int total_cycles = 2;

        SolverParameters params;
        params.verbose = false;
        params.tolerance = 1e-10;
        params.max_iterations = 10000;

        const Problems::ADRProblem<2> problem;
        constexpr int fe_degree = 2;

        for (int i = 0; i < total_cycles; ++i) {
            int r = start_ref + i;
            double h = compute_h(r);

            if (is_root) {
                std::cout << "Processing Refinement Level: " << r << " (h=" << h
                          << ")" << std::endl;
            }

            // ------------------------------------------
            // 1. Run Matrix Based Solver
            // ------------------------------------------
            params.output_prefix = "sol_mb_" + std::to_string(r);
            MatrixBasedSolver solver_mb(problem, fe_degree, MPI_COMM_WORLD,
                                        params);

            // Timer for total wall time
            Timer timer_mb;
            timer_mb.start();

            solver_mb.run(r);
            double error_mb = solver_mb.compute_l2_error();

            timer_mb.stop();

            // ------------------------------------------
            // 2. Run Matrix Free Solver
            // ------------------------------------------
            params.output_prefix = "sol_mf_" + std::to_string(r);
            params.solver_type = SolverType::MatrixFree;
            MatrixFreeSolver<2, fe_degree> solver_mf(problem, MPI_COMM_WORLD,
                                                     params);

            Timer timer_mf;
            timer_mf.start();

            solver_mf.run(r);
            double error_mf = solver_mf.compute_l2_error();

            timer_mf.stop();

            // ------------------------------------------
            // Write Results
            // ------------------------------------------
            if (is_root) {
                csv_file << r << "," << h << "," << timer_mb.wall_time() << ","
                         << error_mb << "," << timer_mf.wall_time() << ","
                         << error_mf << "\n";

                // Flush to ensure data is saved if crash occurs
                csv_file.flush();
            }

            // Barrier to keep clean output
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (is_root) {
            csv_file.close();
            std::cout << "Convergence study complete. Data written to "
                         "convergence_results.csv"
                      << std::endl;
        }

    } catch (std::exception& exc) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    } catch (...) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}