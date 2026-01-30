/**
 * @file main_matrix_based.cpp
 * @brief Main driver for the matrix-based ADR solver with hybrid
 * parallelization.
 *
 * This program demonstrates the matrix-based solver using hybrid MPI+threading
 * parallelization for the Advection-Diffusion-Reaction problem.
 */
#include "core/problem_definition.h"
#include "core/types.h"
#include "matrix_based/matrix_based_solver.h"
#include <deal.II/base/mpi.h>

int main(int argc, char* argv[]) {
    try {
        using namespace dealii;
        using namespace HybridADRSolver;

        Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 8);

        SolverParameters params;
        params.verbose = true;
        params.output_prefix = "solution";

        const Problems::ADRProblem<2> problem;

        MatrixBasedSolver solver(problem, 2, MPI_COMM_WORLD, params);

        solver.run(9);

    } catch (std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
    return 0;
}