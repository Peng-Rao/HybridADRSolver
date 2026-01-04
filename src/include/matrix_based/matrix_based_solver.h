/**
 * @file matrix_based_solver.h
 * @brief Matrix-based solver using WorkStream for hybrid parallelization
 */

#ifndef HYBRIDADRSOLVER_MATRIX_BASED_SOLVER_H
#define HYBRIDADRSOLVER_MATRIX_BASED_SOLVER_H

#include "core/problem_definition.h"
#include "core/solver.h"
#include "core/types.h"

namespace HybridADRSolver {
using namespace dealii;

// Forward declarations
template <int dim> struct ScratchData;
struct CopyData;

/**
 * @brief Matrix-based solver with hybrid MPI+threading parallelization
 * * This solver:
 * - Explicitly assembles and stores the sparse system matrix
 * - Uses WorkStream for thread-parallel assembly
 * - Uses PETSc for distributed linear algebra
 * - Supports ILU and AMG preconditioning
 * * @tparam dim Spatial dimension
 */
template <int dim> class MatrixBasedSolver : public ParallelSolverBase<dim> {
public:
    /**
     * Constructor
     * @param problem The PDE problem to solve
     * @param degree Polynomial degree of finite elements
     * @param comm MPI communicator
     * @param params Solver parameters
     */
    MatrixBasedSolver(const ProblemInterface<dim>& problem, unsigned int degree,
                      MPI_Comm comm,
                      const SolverParameters& params = SolverParameters());

    void run(unsigned int n_refinements) override;

    SolverType get_solver_type() const override {
        return SolverType::MatrixBased;
    }

    std::string get_name() const override { return "Matrix-Based Solver"; }

    /**
     * Get the system matrix (for analysis/debugging)
     */
    const LADistributed::MPI::SparseMatrix& get_system_matrix() const {
        return system_matrix;
    }

    /**
     * @brief Compute memory usage in MB.
     */
    double compute_memory_usage() const;

protected:
    void setup_dofs() override;
    void assemble_system() override;
    void solve() override;
    void output_results(unsigned int cycle) const override;
    double compute_l2_error();

private:
    const ProblemInterface<dim>& problem;
    unsigned int fe_degree;

    LADistributed::MPI::SparseMatrix system_matrix;
    LADistributed::MPI::Vector system_rhs;
    LADistributed::MPI::Vector solution;
};

} // namespace HybridADRSolver

#endif // HYBRIDADRSOLVER_MATRIX_BASED_SOLVER_H
