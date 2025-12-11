#include "basic_solver.h"
#include "problem_parameters.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    // Initialize MPI for deal.II
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    return RUN_ALL_TESTS();
}