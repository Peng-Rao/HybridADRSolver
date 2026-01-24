/**
 * @file matrix_free_solver.h
 * @brief Matrix-free solver with hybrid MPI+threading parallelization.
 */

#ifndef HYBRIDADRSOLVER_MATRIX_FREE_SOLVER_H
#define HYBRIDADRSOLVER_MATRIX_FREE_SOLVER_H

#include "adr_operator.h"
#include "core/problem_definition.h"
#include "core/solver.h"
#include "core/types.h"

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/vector_tools.h>

namespace HybridADRSolver {

using namespace dealii;

/**
 * @brief Matrix-free solver with hybrid MPI+threading parallelization.
 *
 * This class implements a matrix-free finite element solver that:
 * - Uses MatrixFree for efficient operator application
 * - Supports hybrid MPI (distributed) + threading (shared memory)
 * parallelization
 * - Implements GMRES/CG with Jacobi preconditioning
 * - Achieves better memory efficiency and cache utilization than matrix-based
 *
 * @tparam dim Spatial dimension
 * @tparam fe_degree Polynomial degree of finite elements
 */
template <int dim, int fe_degree>
class MatrixFreeSolver : public ParallelSolverBase<dim> {
public:
    using Number = double;
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    /**
     * @brief Constructor.
     *
     * @param problem The PDE problem to solve
     * @param comm MPI communicator
     * @param params Solver parameters
     */
    MatrixFreeSolver(const ProblemInterface<dim>& problem, MPI_Comm comm,
                     const SolverParameters& params = SolverParameters());

    /**
     * @brief Run the complete solve cycle.
     *
     * @param n_refinements Number of global mesh refinements
     */
    void run(unsigned int n_refinements) override;

    /**
     * @brief Returns the solver type identifier.
     */
    SolverType get_solver_type() const override {
        return SolverType::MatrixFree;
    }

    /**
     * @brief Returns a descriptive name for this solver.
     */
    std::string get_name() const override {
        return "Matrix-Free Solver (Hybrid MPI+Threading)";
    }

    /**
     * @brief Get the system operator (for analysis/debugging).
     */
    const ADROperator<dim, fe_degree, Number>& get_system_operator() const {
        return system_operator;
    }

    /**
     * @brief Get the solution vector.
     */
    const VectorType& get_solution() const { return solution; }

    /**
     * @brief Compute memory usage in MB.
     */
    double get_memory_usage() const;

    /**
     * @brief Compute the L2 error against the exact solution.
     */
    double compute_l2_error() const;

protected:
    void setup_dofs() override;
    void assemble_system() override;
    void solve() override;
    void output_results(unsigned int cycle) const override;

    /**
     * @brief Setup the MatrixFree data structure.
     */
    void setup_matrix_free();

    /**
     * @brief Assemble the right-hand side vector.
     */
    void assemble_rhs();

    /**
     * @brief Solve using GMRES with Jacobi preconditioning.
     */
    void solve_gmres_jacobi();

    /**
     * @brief Solve using CG (for symmetric problems).
     */
    void solve_cg();

private:
    const ProblemInterface<dim>& problem;

    // MatrixFree data structure
    std::shared_ptr<MatrixFree<dim, Number>> matrix_free_data;

    // System operator (matrix-free)
    ADROperator<dim, fe_degree, Number> system_operator;

    // Vectors
    VectorType solution;
    VectorType system_rhs;
};

// Explicit instantiation declarations
extern template class MatrixFreeSolver<2, 1>;
extern template class MatrixFreeSolver<2, 2>;
extern template class MatrixFreeSolver<2, 3>;
extern template class MatrixFreeSolver<3, 1>;
extern template class MatrixFreeSolver<3, 2>;
extern template class MatrixFreeSolver<3, 3>;

} // namespace HybridADRSolver

#endif // HYBRIDADRSOLVER_MATRIX_FREE_SOLVER_H
