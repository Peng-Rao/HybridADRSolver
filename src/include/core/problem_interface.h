#ifndef HYBRIDADRSOLVER_PROBLEM_INTERFACE_H
#define HYBRIDADRSOLVER_PROBLEM_INTERFACE_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <memory>
#include <set>

namespace HybridADRSolver {
using namespace dealii;

/**
 * @brief Abstract interface for PDE problem coefficients and data
 *
 * Derived classes implement specific PDEs by providing:
 * - Diffusion coefficient μ(x)
 * - advection field β(x)
 * - Reaction coefficient γ(x)
 * - Source term f(x)
 * - Boundary data g(x), h(x)
 *
 * @tparam dim Spatial dimension
 */
template <int dim> class ProblemInterface {
public:
    virtual ~ProblemInterface() = default;

    /**
     * @brief Get problem name for output/logging
     */
    virtual std::string get_name() const = 0;

    // ========================================================================
    // Scalar coefficient evaluation (for matrix-based assembly)
    // ========================================================================

    /**
     * @brief Evaluate diffusion coefficient μ at point p
     */
    virtual double diffusion(const Point<dim>& p) const = 0;

    /**
     * @brief Evaluate advection field β at point p
     */
    virtual Tensor<1, dim> advection(const Point<dim>& p) const = 0;

    /**
     * @brief Evaluate reaction coefficient γ at point p
     */
    virtual double reaction(const Point<dim>& p) const = 0;

    /**
     * @brief Evaluate source term f at point p
     */
    virtual double source(const Point<dim>& p) const = 0;

    // ========================================================================
    // Vectorized coefficient evaluation (for matrix-free)
    // ========================================================================

    /**
     * @brief Vectorized diffusion coefficient evaluation
     */
    virtual VectorizedArray<double>
    diffusion_vectorized(const Point<dim, VectorizedArray<double>>& p) const {
        VectorizedArray<double> result;
        for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v) {
            Point<dim> point;
            for (unsigned int d = 0; d < dim; ++d)
                point[d] = p[d][v];
            result[v] = diffusion(point);
        }
        return result;
    }

    /**
     * @brief Vectorized advection field evaluation
     */
    virtual Tensor<1, dim, VectorizedArray<double>>
    advection_vectorized(const Point<dim, VectorizedArray<double>>& p) const {
        Tensor<1, dim, VectorizedArray<double>> result;
        for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v) {
            Point<dim> point;
            for (unsigned int d = 0; d < dim; ++d)
                point[d] = p[d][v];
            Tensor<1, dim> beta = advection(point);
            for (unsigned int d = 0; d < dim; ++d)
                result[d][v] = beta[d];
        }
        return result;
    }

    /**
     * @brief Vectorized reaction coefficient evaluation
     */
    virtual VectorizedArray<double>
    reaction_vectorized(const Point<dim, VectorizedArray<double>>& p) const {
        VectorizedArray<double> result;
        for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v) {
            Point<dim> point;
            for (unsigned int d = 0; d < dim; ++d)
                point[d] = p[d][v];
            result[v] = reaction(point);
        }
        return result;
    }

    /**
     * @brief Vectorized source term evaluation
     */
    virtual VectorizedArray<double>
    source_vectorized(const Point<dim, VectorizedArray<double>>& p) const {
        VectorizedArray<double> result;
        for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v) {
            Point<dim> point;
            for (unsigned int d = 0; d < dim; ++d)
                point[d] = p[d][v];
            result[v] = source(point);
        }
        return result;
    }

    // ========================================================================
    // Boundary conditions
    // ========================================================================

    /**
     * @brief Get Dirichlet boundary function
     */
    virtual std::shared_ptr<Function<dim>> get_dirichlet_bc() const = 0;

    /**
     * @brief Get Neumann boundary function
     */
    virtual std::shared_ptr<Function<dim>> get_neumann_bc() const = 0;

    /**
     * @brief Check if the operator is symmetric
     * (No advection term or properly skew-symmetrized)
     */
    virtual bool is_symmetric() const { return false; }

    /**
     * @brief Get Dirichlet boundary IDs
     */
    virtual std::set<types::boundary_id> get_dirichlet_boundary_ids() const {
        return {0}; // Default: boundary id 0
    }

    /**
     * @brief Get Neumann boundary IDs
     */
    virtual std::set<types::boundary_id> get_neumann_boundary_ids() const {
        return {1}; // Default: boundary id 1
    }
};

// ==========================================================================
// Function wrappers for deal.II interface
// ==========================================================================

/**
 * @brief Wrapper to use ProblemInterface coefficients as deal.II Functions
 */
template <int dim> class DiffusionFunction : public Function<dim> {
public:
    explicit DiffusionFunction(const ProblemInterface<dim>& problem)
        : problem(problem) {}

    double value(const Point<dim>& p, const unsigned int = 0) const override {
        return problem.diffusion(p);
    }

private:
    const ProblemInterface<dim>& problem;
};

template <int dim> class ReactionFunction : public Function<dim> {
public:
    explicit ReactionFunction(const ProblemInterface<dim>& problem)
        : problem(problem) {}

    double value(const Point<dim>& p, const unsigned int = 0) const override {
        return problem.reaction(p);
    }

private:
    const ProblemInterface<dim>& problem;
};

template <int dim> class SourceFunction : public Function<dim> {
public:
    explicit SourceFunction(const ProblemInterface<dim>& problem)
        : problem(problem) {}

    double value(const Point<dim>& p, const unsigned int = 0) const override {
        return problem.source(p);
    }

private:
    const ProblemInterface<dim>& problem;
};
} // namespace HybridADRSolver

#endif // HYBRIDADRSOLVER_PROBLEM_INTERFACE_H
