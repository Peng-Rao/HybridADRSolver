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
#include <deal.II/multigrid/mg_constrained_dofs.h>

namespace HybridADRSolver {
using namespace dealii;

/**
 * @brief Matrix-free operator for the ADR problem with multigrid support.
 *
 * This class implements the action of the system matrix on a vector without
 * explicitly storing the matrix. The computation uses:
 * - Sum factorization for efficient tensor-product evaluation
 * - Vectorization over multiple cells (SIMD)
 * - Hybrid MPI+threading parallelization
 * - Geometric Multigrid (GMG) preconditioning support
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
class ADROperator : public MatrixFreeOperators::Base<
                        dim, LinearAlgebra::distributed::Vector<Number>> {
public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using value_type = Number;
    using size_type = types::global_dof_index;
    using Base =
        MatrixFreeOperators::Base<dim,
                                  LinearAlgebra::distributed::Vector<Number>>;

    /**
     * @brief Default constructor.
     */
    ADROperator() : problem_ptr(nullptr), is_mg_level(false), mg_level(-1) {}

    /**
     * @brief Initialize the operator with MatrixFree data and problem
     * definition.
     *
     * @param data_in Shared pointer to MatrixFree data structure
     * @param problem Reference to the problem interface
     */
    void initialize(std::shared_ptr<const MatrixFree<dim, Number>> data_in,
                    const ProblemInterface<dim>& problem) {
        Base::initialize(data_in);
        this->problem_ptr = &problem;
        this->is_mg_level = false;
        this->mg_level = -1;
        precompute_coefficient_data();
    }

    /**
     * @brief Initialize for a multigrid level.
     *
     * @param data_in Shared pointer to MatrixFree data structure for this level
     * @param mg_constrained_dofs The MGConstrainedDoFs object
     * @param level The multigrid level
     * @param problem Reference to the problem interface
     */
    void initialize(std::shared_ptr<const MatrixFree<dim, Number>> data_in,
                    const MGConstrainedDoFs& mg_constrained_dofs,
                    unsigned int level, const ProblemInterface<dim>& problem) {
        Base::initialize(data_in, mg_constrained_dofs, level);
        this->problem_ptr = &problem;
        this->is_mg_level = true;
        this->mg_level = level;
        precompute_coefficient_data();
    }

    /**
     * @brief Clear all data.
     */
    void clear() override {
        Base::clear();
        problem_ptr = nullptr;
        diffusion_coefficients.clear();
        advection_coefficients.clear();
        reaction_coefficients.clear();
        is_mg_level = false;
        mg_level = -1;
    }

    /**
     * @brief Computes the diagonal of the operator (for preconditioning).
     *
     * Uses MatrixFreeTools for efficient diagonal computation.
     */
    void compute_diagonal() override {
        this->inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
        VectorType& inverse_diagonal =
            this->inverse_diagonal_entries->get_vector();
        this->data->initialize_dof_vector(inverse_diagonal);

        MatrixFreeTools::compute_diagonal(*this->data, inverse_diagonal,
                                          &ADROperator::local_compute_diagonal,
                                          this);

        this->set_constrained_entries_to_one(inverse_diagonal);

        for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size();
             ++i) {
            if (std::abs(inverse_diagonal.local_element(i)) > 1e-14)
                inverse_diagonal.local_element(i) =
                    1.0 / inverse_diagonal.local_element(i);
            else
                inverse_diagonal.local_element(i) = 1.0;
        }
    }

    /**
     * @brief Access to the underlying MatrixFree object.
     */
    std::shared_ptr<const MatrixFree<dim, Number>>
    get_matrix_free() const override {
        return this->data;
    }

    /**
     * @brief Check if this is a multigrid level operator.
     */
    [[nodiscard]] bool is_multigrid_level() const { return is_mg_level; }

    /**
     * @brief Get the multigrid level (-1 if not a level operator).
     */
    [[nodiscard]] int get_mg_level() const { return mg_level; }

private:
    /**
     * @brief Applies the operator: dst += A * src
     *
     * This is called by the base class vmult/vmult_add methods.
     */
    void apply_add(VectorType& dst, const VectorType& src) const override {
        this->data->cell_loop(&ADROperator::local_apply, this, dst, src);
    }

    /**
     * @brief Precomputes coefficient values at quadrature points.
     *
     * This avoids repeated evaluation of coefficient functions during
     * each operator application, which is especially important for
     * non-constant coefficients.
     */
    void precompute_coefficient_data() {
        Assert(problem_ptr != nullptr, ExcNotInitialized());

        const unsigned int n_cells = this->data->n_cell_batches();
        const unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);

        diffusion_coefficients.resize(n_cells);
        advection_coefficients.resize(n_cells);
        reaction_coefficients.resize(n_cells);

        // Use FEEvaluation to get quadrature points
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(*this->data);

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
    local_apply(const MatrixFree<dim, Number>& mf_data, VectorType& dst,
                const VectorType& src,
                const std::pair<unsigned int, unsigned int>& cell_range) const {
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(mf_data);

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
     * @brief Local diagonal computation for MatrixFreeTools::compute_diagonal.
     *
     * This function is called by MatrixFreeTools to compute the diagonal
     * entries efficiently.
     */
    void local_compute_diagonal(
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>& phi) const {
        const unsigned int cell = phi.get_current_cell_index();

        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        for (unsigned int q = 0; q < phi.n_q_points; ++q) {
            const VectorizedArray<Number> u_val = phi.get_value(q);
            const Tensor<1, dim, VectorizedArray<Number>> grad_u =
                phi.get_gradient(q);

            // Diffusion
            Tensor<1, dim, VectorizedArray<Number>> flux =
                diffusion_coefficients[cell][q] * grad_u;

            // Advection
            VectorizedArray<Number> advection_val =
                advection_coefficients[cell][q] * grad_u;

            // Reaction
            VectorizedArray<Number> val =
                reaction_coefficients[cell][q] * u_val + advection_val;

            phi.submit_gradient(flux, q);
            phi.submit_value(val, q);
        }

        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    }

    // Data members
    const ProblemInterface<dim>* problem_ptr;
    bool is_mg_level;
    unsigned int mg_level;

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
        op.compute_diagonal();
        inverse_diagonal = op.get_matrix_diagonal_inverse()->get_vector();
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