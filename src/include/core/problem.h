/**
 * @file problem.h
 * @brief Advection-Diffusion-Reaction problem definition
 *
 * Implements Problem (8):
 *   -∇·(μ∇u) + ∇·(βu) + γu = f   in Ω
 *   u = g                         on Γ_D
 *   ∇u·n = h                      on Γ_N
 */

#ifndef HYBRIDADRSOLVER_PROBLEM_H
#define HYBRIDADRSOLVER_PROBLEM_H

#include "core/problem_interface.h"

namespace HybridADRSolver::Problems {
using namespace dealii;

/**
 * @brief Standard CDR problem with rotating advection field
 *
 * Coefficients:
 * - μ(x) = 1 + 0.5*sin(πx)  (variable diffusion)
 * - β(x) = (-y, x)          (rotating advection)
 * - γ(x) = 1                (constant reaction)
 * - f(x) = exp(-10r²)       (Gaussian source)
 */
template <int dim> class ADRProblem : public ProblemInterface<dim> {
public:
    ADRProblem() = default;

    std::string get_name() const override {
        return "advection-Diffusion-Reaction";
    }

    double diffusion(const Point<dim>& p) const override {
        return 1.0 + 0.5 * std::sin(numbers::PI * p[0]);
    }

    Tensor<1, dim> advection(const Point<dim>& p) const override {
        Tensor<1, dim> beta;
        beta[0] = -p[1];
        beta[1] = p[0];
        if (dim == 3)
            beta[2] = 0.0;
        return beta;
    }

    double reaction(const Point<dim>& p) const override {
        (void)p;
        return 1.0;
    }

    double source(const Point<dim>& p) const override {
        return std::exp(-10.0 * p.norm_square());
    }
    double exact_solution(const Point<dim>& /*p*/) const override {
        return 0.0;
    }

    // Optimized vectorized versions
    VectorizedArray<double> diffusion_vectorized(
        const Point<dim, VectorizedArray<double>>& p) const override {
        VectorizedArray<double> result;
        for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
            result[v] = 1.0 + 0.5 * std::sin(numbers::PI * p[0][v]);
        return result;
    }

    Tensor<1, dim, VectorizedArray<double>> advection_vectorized(
        const Point<dim, VectorizedArray<double>>& p) const override {
        Tensor<1, dim, VectorizedArray<double>> beta;
        beta[0] = -p[1];
        beta[1] = p[0];
        if (dim == 3)
            beta[2] = VectorizedArray<double>(0.0);
        return beta;
    }

    VectorizedArray<double> reaction_vectorized(
        const Point<dim, VectorizedArray<double>>&) const override {
        return VectorizedArray<double>(1.0);
    }

    VectorizedArray<double> source_vectorized(
        const Point<dim, VectorizedArray<double>>& p) const override {
        VectorizedArray<double> r_sq = VectorizedArray<double>(0.0);
        for (unsigned int d = 0; d < dim; ++d)
            r_sq += p[d] * p[d];

        VectorizedArray<double> result;
        for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
            result[v] = std::exp(-10.0 * r_sq[v]);
        return result;
    }

    std::shared_ptr<Function<dim>> get_dirichlet_bc() const override {
        return std::make_shared<Functions::ZeroFunction<dim>>();
    }

    std::shared_ptr<Function<dim>> get_neumann_bc() const override {
        return std::make_shared<Functions::ZeroFunction<dim>>();
    }

    bool is_symmetric() const override {
        return false; // Has advection term
    }
};

/**
 * @brief Pure diffusion (Laplace) problem for comparison
 */
template <int dim> class LaplaceProblem : public ProblemInterface<dim> {
public:
    std::string get_name() const override { return "Laplace (Pure Diffusion)"; }

    double diffusion(const Point<dim>&) const override { return 1.0; }

    Tensor<1, dim> advection(const Point<dim>&) const override {
        return Tensor<1, dim>(); // Zero advection
    }

    double reaction(const Point<dim>&) const override {
        return 0.0; // No reaction
    }

    double source(const Point<dim>& p) const override {
        // f = 2*dim for solution u = x*(1-x) + y*(1-y) + ...
        return 2.0 * dim;
    }

    /**
     * @brief Analytical solution for verification.
     * u(x) = x(1-x) + y(1-y) ...
     */
    double exact_solution(const Point<dim>& p) const override {
        double val = 0.0;
        for (unsigned int d = 0; d < dim; ++d) {
            val += p[d] * (1.0 - p[d]);
        }
        return val;
    }

    VectorizedArray<double> diffusion_vectorized(
        const Point<dim, VectorizedArray<double>>&) const override {
        return VectorizedArray<double>(1.0);
    }

    Tensor<1, dim, VectorizedArray<double>> advection_vectorized(
        const Point<dim, VectorizedArray<double>>&) const override {
        return Tensor<1, dim, VectorizedArray<double>>();
    }

    VectorizedArray<double> reaction_vectorized(
        const Point<dim, VectorizedArray<double>>&) const override {
        return VectorizedArray<double>(0.0);
    }

    VectorizedArray<double> source_vectorized(
        const Point<dim, VectorizedArray<double>>&) const override {
        return VectorizedArray<double>(2.0 * dim);
    }

    std::shared_ptr<Function<dim>> get_dirichlet_bc() const override {
        return std::make_shared<Functions::ZeroFunction<dim>>();
    }

    std::shared_ptr<Function<dim>> get_neumann_bc() const override {
        return std::make_shared<Functions::ZeroFunction<dim>>();
    }

    bool is_symmetric() const override {
        return true; // No advection
    }
};

/**
 * @brief advection-dominated problem (high Peclet number)
 */
template <int dim>
class advectionDominatedProblem : public ProblemInterface<dim> {
public:
    explicit advectionDominatedProblem(double peclet = 100.0)
        : peclet_number(peclet) {}

    std::string get_name() const override {
        return "advection-Dominated (Pe=" +
               std::to_string(static_cast<int>(peclet_number)) + ")";
    }

    double diffusion(const Point<dim>&) const override {
        return 1.0 / peclet_number; // Small diffusion
    }

    Tensor<1, dim> advection(const Point<dim>&) const override {
        Tensor<1, dim> beta;
        beta[0] = 1.0; // Uniform advection in x
        return beta;
    }

    double reaction(const Point<dim>&) const override { return 0.0; }

    double source(const Point<dim>&) const override { return 1.0; }

    std::shared_ptr<Function<dim>> get_dirichlet_bc() const override {
        return std::make_shared<Functions::ZeroFunction<dim>>();
    }

    std::shared_ptr<Function<dim>> get_neumann_bc() const override {
        return std::make_shared<Functions::ZeroFunction<dim>>();
    }

    bool is_symmetric() const override { return false; }

private:
    double peclet_number;
};

} // namespace HybridADRSolver::Problems

#endif // HYBRIDADRSOLVER_PROBLEM_H
