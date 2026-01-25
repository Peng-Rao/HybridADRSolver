/**
 * @file main_matrix_free.cpp
 * @brief Main driver for the matrix-free ADR solver with hybrid
 * parallelization.
 *
 * This program demonstrates the matrix-free solver using hybrid MPI+threading
 * parallelization for the Advection-Diffusion-Reaction problem.
 */

#include "core/problem_definition.h"
#include "core/types.h"
#include "matrix_free/matrix_free_solver.h"
#include <deal.II/base/mpi.h>

int main(int argc, char* argv[]) {
    try {
        using namespace dealii;
        using namespace HybridADRSolver;

        // Initialize MPI with threading support
        // The second argument (1) means we want 1 thread per MPI process
        // initially but this can be overridden by TBB/threading settings
        Utilities::MPI::MPI_InitFinalize mpi(argc, argv,
                                             numbers::invalid_unsigned_int);

        // Configure solver parameters
        SolverParameters params;
        params.verbose = true;
        params.output_prefix = "solution_mf";
        params.solver_type = SolverType::MatrixFree;
        params.max_iterations = 1000;
        params.tolerance = 1e-10;

        // Define the problem
        const Problems::ADRProblem<2> problem;

        // Create and run the matrix-free solver with polynomial degree 2
        constexpr int fe_degree = 2;
        MatrixFreeSolver<2, fe_degree> solver(problem, MPI_COMM_WORLD, params);

        // Run with 4 global refinements
        solver.run(4);

    } catch (std::exception& exc) {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception!" << std::endl;
        return 1;
    }

    return 0;
}
