#include "basic_solver.h"
#include "problem_parameters.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    testing::InitGoogleTest(&argc, argv);
    const unsigned int my_rank =
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    testing::TestEventListeners& listeners =
        testing::UnitTest::GetInstance()->listeners();

    if (my_rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    return RUN_ALL_TESTS();
}

TEST(AdvectionDiffusionTest, ConvergenceCheck2D) {
    constexpr int dim = 2;
    constexpr int fe_degree = 1;

    BasicSolver::Solver<dim> solver(fe_degree);

    testing::internal::CaptureStdout();
    solver.run();
    std::string output = testing::internal::GetCapturedStdout();

    const double error = solver.compute_l2_error();

    EXPECT_LT(error, 0.05) << "L2 Error is too high!";

    std::cout << "Computed L2 Error: " << error << std::endl;
}

TEST(AdvectionDiffusionTest, ParameterCheck) {
    constexpr int dim = 2;
    constexpr dealii::Point<dim> p(0.5, 0.5);

    const parameters::DiffusionCoefficient<dim> mu(1e-5);
    const parameters::AdvectionField<dim> beta;

    const double mu_val = mu.value(p, 0);
    const dealii::Tensor<1, dim> beta_val = beta.value(p);

    EXPECT_DOUBLE_EQ(mu_val, 1e-5);
    EXPECT_NEAR(beta_val.norm(), std::sqrt(0.5 * 0.5 + 0.5 * 0.5), 1e-6);
    const double h = 0.1;
    const double Pe = beta_val.norm() * h / (2 * mu_val);
    EXPECT_GT(Pe, 1.0) << "Should be advection dominated for these parameters";
}