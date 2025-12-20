/**
 * @file main.cpp
 * @brief Main entry point for the Hybrid ADR Solver application.
 */

#include "core/problem.h"
#include "core/types.h"
#include "matrix_based/matrix_based_solver.h"

#include <deal.II/base/mpi.h>

int main(int argc, char* argv[]) {
    try {
        using namespace dealii;
        using namespace HybridADRSolver;
        // 1. Initialize MPI
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        MPI_Comm mpi_communicator = MPI_COMM_WORLD;

        // 2. Setup Solver Parameters
        SolverParameters params;
        params.verbose = true;
        params.max_iterations = 1000;
        params.tolerance = 1e-10;
        params.output_solution = true;
        params.output_prefix = "solution";

        // 3. Define dimension and refinement level
        constexpr int dim = 2;
        constexpr unsigned int n_global_refinements = 4;
        constexpr unsigned int fe_degree = 1;

        // 4. Instantiate Problem
        Problems::ADRProblem<dim> problem;

        // 5. Instantiate Solver
        // Note: We use the abstract base pointer to demonstrate polymorphism,
        // though stack allocation is fine too.
        const std::unique_ptr<ParallelSolverBase<dim>> solver =
            std::make_unique<MatrixBasedSolver<dim>>(problem, fe_degree,
                                                     mpi_communicator, params);

        // 6. Run Simulation
        solver->run(n_global_refinements);

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
    }
    return 0;
}
