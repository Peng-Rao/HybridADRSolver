#ifndef HYBRIDADRSOLVER_ADR_OPERATOR_H
#define HYBRIDADRSOLVER_ADR_OPERATOR_H

#include "core/problem_definition.h"
#include "core/types.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

namespace HybridADRSolver {
using namespace dealii;

/**
 * @brief Matrix-free operator for the ADR problem.
 *
 * This class implements the action of the system matrix on a vector without
 * explicitly storing the matrix. The computation uses:
 * - Sum factorization for efficient tensor-product evaluation
 * - Vectorization over multiple cells (SIMD)
 * - Hybrid MPI+threading parallelization
 *
 * The weak form implemented is:
 * \f$ a(u,v) = (\mu \nabla u, \nabla v) + (\beta \cdot \nabla u, v) + (\gamma
 * u, v) \f$
 *
 * @tparam dim Spatial dimension
 * @tparam fe_degree Polynomial degree of finite elements
 * @tparam Number Floating point type (double or float)
 */
template <int dim, int fe_degree, typename Number = double>
class ADROperator : public Subscriptor {
public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using value_type = Number;
    using size_type = types::global_dof_index;

    /**
     * @brief Default constructor.
     */
    ADROperator() : problem_ptr(nullptr) {}

    /**
     * @brief Initialize the operator with MatrixFree data and problem
     * definition.
     *
     * @param data_in Shared pointer to MatrixFree data structure
     * @param problem Reference to the problem interface
     */
    void initialize(std::shared_ptr<const MatrixFree<dim, Number>> data_in,
                    const ProblemInterface<dim>& problem) {
        this->data = data_in;
        this->problem_ptr = &problem;
        precompute_coefficient_data();
    }

    /**
     * @brief Clear all data.
     */
    void clear() {
        data.reset();
        problem_ptr = nullptr;
        diffusion_coefficients.clear();
        advection_coefficients.clear();
        reaction_coefficients.clear();
    }

    /**
     * @brief Returns the number of rows (degrees of freedom).
     */
    size_type m() const {
        Assert(data.get() != nullptr, ExcNotInitialized());
        return data->get_dof_handler().n_dofs();
    }

    /**
     * @brief Returns the number of columns (equals rows for square operator).
     */
    size_type n() const { return m(); }

    /**
     * @brief Applies the operator: dst = A * src
     *
     * @param dst Output vector
     * @param src Input vector
     */
    void vmult(VectorType& dst, const VectorType& src) const {
        Assert(data.get() != nullptr, ExcNotInitialized());
        dst = 0;
        vmult_add(dst, src);
    }

    /**
     * @brief Adds the operator application: dst += A * src
     *
     * @param dst Output vector (accumulated)
     * @param src Input vector
     */
    void vmult_add(VectorType& dst, const VectorType& src) const {
        Assert(data.get() != nullptr, ExcNotInitialized());
        data->cell_loop(&ADROperator::local_apply, this, dst, src);
    }

    /**
     * @brief Applies the transpose operator: dst = A^T * src
     *
     * For the ADR problem with advection, the operator is non-symmetric,
     * so Tvmult differs from vmult.
     */
    void Tvmult(VectorType& dst, const VectorType& src) const {
        Assert(data.get() != nullptr, ExcNotInitialized());
        dst = 0;
        Tvmult_add(dst, src);
    }

    /**
     * @brief Adds the transpose operator application: dst += A^T * src
     */
    void Tvmult_add(VectorType& dst, const VectorType& src) const {
        Assert(data.get() != nullptr, ExcNotInitialized());
        // For the transpose, we negate the advection term
        data->cell_loop(&ADROperator::local_apply_transpose, this, dst, src);
    }

    /**
     * @brief Computes the diagonal of the operator (for preconditioning).
     *
     * @param diagonal Output vector containing the diagonal entries
     */
    void compute_diagonal(VectorType& diagonal) const {
        Assert(data.get() != nullptr, ExcNotInitialized());

        diagonal.reinit(data->get_dof_handler().locally_owned_dofs(),
                        data->get_task_info().communicator);

        // Use FEEvaluation to compute diagonal entries
        data->initialize_dof_vector(diagonal);
        diagonal = 0;

        // Apply to unit vectors to extract diagonal
        VectorType ones;
        data->initialize_dof_vector(ones);

        // Compute diagonal using cell loop with special diagonal computation
        data->cell_loop(&ADROperator::local_compute_diagonal, this, diagonal,
                        ones);

        diagonal.compress(VectorOperation::add);
    }

    /**
     * @brief Returns the inverse of the diagonal (for Jacobi preconditioning).
     *
     * @param inverse_diagonal Output vector
     */
    void compute_inverse_diagonal(VectorType& inverse_diagonal) const {
        compute_diagonal(inverse_diagonal);

        // Invert diagonal entries, handling near-zero values
        for (auto& val : inverse_diagonal)
            val = (std::abs(val) > 1e-14) ? 1.0 / val : 1.0;
    }

    /**
     * @brief Access to the underlying MatrixFree object.
     */
    std::shared_ptr<const MatrixFree<dim, Number>> get_matrix_free() const {
        return data;
    }

    /**
     * @brief Initialize a vector compatible with this operator.
     */
    void initialize_dof_vector(VectorType& vec) const {
        data->initialize_dof_vector(vec);
    }

private:
    /**
     * @brief Precomputes coefficient values at quadrature points.
     *
     * This avoids repeated evaluation of coefficient functions during
     * each operator application, which is especially important for
     * non-constant coefficients.
     */
    void precompute_coefficient_data() {
        Assert(problem_ptr != nullptr, ExcNotInitialized());

        const unsigned int n_cells = data->n_cell_batches();
        const unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);

        diffusion_coefficients.resize(n_cells);
        advection_coefficients.resize(n_cells);
        reaction_coefficients.resize(n_cells);

        // Use FEEvaluation to get quadrature points
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(*data);

        for (unsigned int cell = 0; cell < n_cells; ++cell) {
            phi.reinit(cell);

            diffusion_coefficients[cell].resize(n_q_points);
            advection_coefficients[cell].resize(n_q_points);
            reaction_coefficients[cell].resize(n_q_points);

            for (unsigned int q = 0; q < n_q_points; ++q) {
                // Get the vectorized quadrature point
                Point<dim, VectorizedArray<Number>> q_point =
                    phi.quadrature_point(q);

                // Evaluate coefficients for each lane in the vector
                VectorizedArray<Number> mu;
                Tensor<1, dim, VectorizedArray<Number>> beta;
                VectorizedArray<Number> gamma;

                for (unsigned int v = 0; v < VectorizedArray<Number>::size();
                     ++v) {
                    Point<dim> p;
                    for (unsigned int d = 0; d < dim; ++d)
                        p[d] = q_point[d][v];

                    mu[v] = problem_ptr->diffusion_coefficient(p);
                    gamma[v] = problem_ptr->reaction_coefficient(p);

                    Tensor<1, dim> beta_scalar =
                        problem_ptr->advection_field(p);
                    for (unsigned int d = 0; d < dim; ++d)
                        beta[d][v] = beta_scalar[d];
                }

                diffusion_coefficients[cell][q] = mu;
                advection_coefficients[cell][q] = beta;
                reaction_coefficients[cell][q] = gamma;
            }
        }
    }

    /**
     * @brief Local cell operation for vmult.
     *
     * This is the core computation kernel that applies the operator
     * on a batch of cells using vectorization (SIMD).
     */
    void
    local_apply(const MatrixFree<dim, Number>& data, VectorType& dst,
                const VectorType& src,
                const std::pair<unsigned int, unsigned int>& cell_range) const {
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell) {
            phi.reinit(cell);
            phi.gather_evaluate(src, EvaluationFlags::values |
                                         EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q) {
                const VectorizedArray<Number> u_val = phi.get_value(q);
                const Tensor<1, dim, VectorizedArray<Number>> grad_u =
                    phi.get_gradient(q);

                // Diffusion: mu * grad(u)
                Tensor<1, dim, VectorizedArray<Number>> flux =
                    diffusion_coefficients[cell][q] * grad_u;

                // Advection contribution to flux: (beta · grad(u)) * v
                // This is handled via the value term below
                VectorizedArray<Number> advection_val =
                    advection_coefficients[cell][q] * grad_u;

                // Reaction: gamma * u
                VectorizedArray<Number> val =
                    reaction_coefficients[cell][q] * u_val + advection_val;

                phi.submit_gradient(flux, q);
                phi.submit_value(val, q);
            }

            phi.integrate_scatter(
                EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
    }

    /**
     * @brief Local cell operation for transpose vmult.
     *
     * The transpose reverses the sign of the advection term.
     */
    void local_apply_transpose(
        const MatrixFree<dim, Number>& data, VectorType& dst,
        const VectorType& src,
        const std::pair<unsigned int, unsigned int>& cell_range) const {
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell) {
            phi.reinit(cell);
            phi.gather_evaluate(src, EvaluationFlags::values |
                                         EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q) {
                const VectorizedArray<Number> u_val = phi.get_value(q);
                const Tensor<1, dim, VectorizedArray<Number>> grad_u =
                    phi.get_gradient(q);

                // Diffusion (symmetric)
                Tensor<1, dim, VectorizedArray<Number>> flux =
                    diffusion_coefficients[cell][q] * grad_u;

                // Transpose of advection: -(beta · grad(v)) * u
                // In weak form, this becomes +(div(beta) * u, v) - (beta * u,
                // grad(v)) For divergence-free beta, this simplifies
                VectorizedArray<Number> advection_val =
                    -advection_coefficients[cell][q] * grad_u;

                // Reaction (symmetric)
                VectorizedArray<Number> val =
                    reaction_coefficients[cell][q] * u_val + advection_val;

                phi.submit_gradient(flux, q);
                phi.submit_value(val, q);
            }

            phi.integrate_scatter(
                EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
    }

    /**
     * @brief Computes diagonal entries using sum factorization.
     */
    void local_compute_diagonal(
        const MatrixFree<dim, Number>& data, VectorType& dst,
        const VectorType& /*src*/,
        const std::pair<unsigned int, unsigned int>& cell_range) const {
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);

        AlignedVector<VectorizedArray<Number>> diagonal_values(
            phi.dofs_per_cell);

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell) {
            phi.reinit(cell);

            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i) {
                // Set unit vector
                for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                    phi.submit_dof_value(VectorizedArray<Number>(), j);
                phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);

                phi.evaluate(EvaluationFlags::values |
                             EvaluationFlags::gradients);

                for (unsigned int q = 0; q < phi.n_q_points; ++q) {
                    const VectorizedArray<Number> u_val = phi.get_value(q);
                    const Tensor<1, dim, VectorizedArray<Number>> grad_u =
                        phi.get_gradient(q);

                    Tensor<1, dim, VectorizedArray<Number>> flux =
                        diffusion_coefficients[cell][q] * grad_u;

                    VectorizedArray<Number> advection_val =
                        advection_coefficients[cell][q] * grad_u;

                    VectorizedArray<Number> val =
                        reaction_coefficients[cell][q] * u_val + advection_val;

                    phi.submit_gradient(flux, q);
                    phi.submit_value(val, q);
                }

                phi.integrate(EvaluationFlags::values |
                              EvaluationFlags::gradients);
                diagonal_values[i] = phi.get_dof_value(i);
            }

            for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
                phi.submit_dof_value(diagonal_values[i], i);
            phi.distribute_local_to_global(dst);
        }
    }

    // Data members
    std::shared_ptr<const MatrixFree<dim, Number>> data;
    const ProblemInterface<dim>* problem_ptr;

    // Precomputed coefficients at quadrature points
    std::vector<std::vector<VectorizedArray<Number>>> diffusion_coefficients;
    std::vector<std::vector<Tensor<1, dim, VectorizedArray<Number>>>>
        advection_coefficients;
    std::vector<std::vector<VectorizedArray<Number>>> reaction_coefficients;
};

/**
 * @brief Jacobi preconditioner for the matrix-free operator.
 *
 * This simple preconditioner uses the inverse of the diagonal entries.
 */
template <int dim, int fe_degree, typename Number = double>
class JacobiPreconditioner {
public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    /**
     * @brief Initialize with the operator.
     */
    void initialize(const ADROperator<dim, fe_degree, Number>& op) {
        op.compute_inverse_diagonal(inverse_diagonal);
    }

    /**
     * @brief Apply the preconditioner: dst = M^{-1} * src
     */
    void vmult(VectorType& dst, const VectorType& src) const {
        dst = src;
        dst.scale(inverse_diagonal);
    }

private:
    VectorType inverse_diagonal;
};
} // namespace HybridADRSolver

#endif // HYBRIDADRSOLVER_ADR_OPERATOR_H
