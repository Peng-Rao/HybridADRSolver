/**
 * @file types.h
 * @brief Common type definitions for the hybrid solver framework
 *
 * This file contains type definitions and aliases used throughout the Hybrid
 * ADR Solver project.
 */

#ifndef HYBRIDADRSOLVER_TYPES_H
#define HYBRIDADRSOLVER_TYPES_H

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace HybridADRSolver {
/**
 * Namespace for distributed linear algebra types (PETSc-based)
 */
namespace LADistributed {
using namespace dealii::LinearAlgebraPETSc;
}

/**
 * Namespace for matrix-free vector types
 */
namespace LAMatrixFree {
template <typename Number = double>
using Vector = dealii::LinearAlgebra::distributed::Vector<Number>;
}

/**
 * Enum for solver type selection
 */
enum class SolverType { MatrixBased, MatrixFree };

/**
 * Enum for preconditioner type
 */
enum class PreconditionerType {
    None,      // No preconditioning
    Jacobi,    // Diagonal/Jacobi preconditioning
    ILU,       // Incomplete LU (matrix-based only)
    AMG,       // Algebraic Multigrid (matrix-based only)
    Chebyshev, // Chebyshev polynomial preconditioning
    GMG        // Geometric Multigrid (matrix-free)
};

/**
 * Enum for linear solver type
 */
enum class LinearSolverType {
    CG,      // Conjugate Gradient (symmetric positive definite)
    GMRES,   // Generalized Minimal Residual (general)
    BiCGStab // BiConjugate Gradient Stabilized
};

/**
 * Structure for solver parameters
 */
struct SolverParameters {
    SolverType solver_type = SolverType::MatrixFree;
    PreconditionerType preconditioner = PreconditionerType::GMG;
    LinearSolverType linear_solver = LinearSolverType::GMRES;
    // Multigrid settings
    bool enable_multigrid = true;        // Enable GMG preconditioning
    unsigned int mg_smoother_degree = 5; // Chebyshev smoother degree
    double mg_smoothing_range = 15.0;    // Smoothing range

    // Linear solver settings
    unsigned int max_iterations = 1000;
    double tolerance = 1e-10;

    // Threading parameters
    unsigned int n_threads = dealii::numbers::invalid_unsigned_int;

    // Output options
    bool verbose = true;
    bool output_solution = true;
    std::string output_prefix = "solution";
};

/**
 * Structure for timing results
 */
struct TimingResults {
    double setup_time = 0.0;
    double assembly_time = 0.0;
    double solve_time = 0.0;
    double total_time = 0.0;
    double operator_apply_time = 0.0;

    unsigned int n_iterations = 0;
    double memory_mb = 0.0;
    unsigned int n_dofs = 0;

    unsigned int n_cells = 0;
    double l2_error = 0.0;
};

} // namespace HybridADRSolver

#endif // HYBRIDADRSOLVER_TYPES_H