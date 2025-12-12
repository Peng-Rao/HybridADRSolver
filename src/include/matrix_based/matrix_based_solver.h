/**
 * @file matrix_based_solver.h
 * @brief Matrix-based solver using WorkStream for hybrid parallelization
 */

#ifndef HYBRIDADRSOLVER_MATRIX_BASED_SOLVER_H
#define HYBRIDADRSOLVER_MATRIX_BASED_SOLVER_H

#include "core/problem_interface.h"
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

    double compute_l2_error();

    /**
     * Get the system matrix (for analysis/debugging)
     */
    const LADistributed::MPI::SparseMatrix& get_system_matrix() const {
        return system_matrix;
    }

protected:
    void setup_dofs() override;
    void assemble_system() override;
    void solve() override;
    void output_results(const unsigned int cycle) const override;

private:
    // WorkStream assembly functions
    void local_assemble_cell(
        const typename DoFHandler<dim>::active_cell_iterator& cell,
        ScratchData<dim>& scratch, CopyData& copy_data);

    void copy_local_to_global(const CopyData& copy_data);

    // Problem definition
    const ProblemInterface<dim>& problem;
    unsigned int fe_degree;

    // Linear algebra objects (Trilinos-based distributed)
    LADistributed::MPI::SparseMatrix system_matrix;
    LADistributed::MPI::Vector system_rhs;
    LADistributed::MPI::Vector solution;
};

} // namespace HybridADRSolver

#endif // HYBRIDADRSOLVER_MATRIX_BASED_SOLVER_H
