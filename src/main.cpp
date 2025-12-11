//
// Created by PENG RAO on 11/12/25.
//
// main.cc
#include "GalerkinSolver.h"

int main(int argc, char* argv[]) {
    try {
        using namespace dealii;
        using namespace GalerkinSolver;

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        Solver<2> problem_2d(1);
        problem_2d.run();

    } catch (std::exception& exc) {
        std::cerr << std::endl
                  << "--------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl;
        return 1;
    } catch (...) {
        std::cerr << std::endl
                  << "--------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl;
        return 1;
    }

    return 0;
}