/**
 * @file problem_definition.h
 * @brief Defines the problem interface and a concrete Advection-Diffusion-Reaction (ADR) problem.
 *
 * This file contains the abstract base class `ProblemInterface` which allows the solver
 * to be decoupled from specific physics. It also provides a concrete implementation
 * `ADRProblem` using the Method of Manufactured Solutions (MMS) for verification.
 */

#ifndef HYBRIDADRSOLVER_PROBLEM_DEFINITION_H
#define HYBRIDADRSOLVER_PROBLEM_DEFINITION_H

#ifndef HYBRIDADRSOLVER_PROBLEM_BASE_H
#define HYBRIDADRSOLVER_PROBLEM_BASE_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <set>
#include <cmath>
#include <string>

namespace HybridADRSolver {
using namespace dealii;
using numbers::PI;

/**
 * @brief Abstract interface for defining a physical problem.
 *
 * This class defines the necessary interfaces for coefficients (diffusion, advection, reaction),
 * source terms, boundary conditions, and exact solutions. The solver relies solely on this
 * interface, allowing for dependency inversion.
 *
 * @tparam dim Spatial dimension of the problem.
 */
template <int dim>
class ProblemInterface {
public:
    virtual ~ProblemInterface() = default;

    /**
     * @brief Returns the diffusion coefficient \f$ \mu(\mathbf{x}) \f$.
     */
    virtual double diffusion_coefficient(const Point<dim>& p) const = 0;

    /**
     * @brief Returns the advection velocity field \f$ \mathbf{\beta}(\mathbf{x}) \f$.
     */
    virtual Tensor<1, dim> advection_field(const Point<dim>& p) const = 0;

    /**
     * @brief Returns the reaction coefficient \f$ \gamma(\mathbf{x}) \f$.
     */
    virtual double reaction_coefficient(const Point<dim>& p) const = 0;

    /**
     * @brief Returns the source term \f$ f(\mathbf{x}) \f$.
     */
    virtual double source_term(const Point<dim>& p) const = 0;

    /**
     * @brief Returns the set of boundary IDs where Dirichlet conditions are applied.
     */
    [[nodiscard]] virtual std::set<types::boundary_id> get_dirichlet_ids() const = 0;

    /**
     * @brief Returns the set of boundary IDs where Neumann conditions are applied.
     */
    [[nodiscard]] virtual std::set<types::boundary_id> get_neumann_ids() const = 0;

    /**
     * @brief Gets the function describing the Dirichlet boundary value \f$ g_D(\mathbf{x}) \f$ for a specific boundary ID.
     */
    virtual const Function<dim>& get_dirichlet_function(types::boundary_id id) const = 0;

    /**
     * @brief Gets the function describing the Neumann flux \f$ g_N(\mathbf{x}) = \nabla u \cdot \mathbf{n} \f$ for a specific boundary ID.
     */
    virtual const Function<dim>& get_neumann_function(types::boundary_id id) const = 0;

    /**
     * @brief Checks if an analytical exact solution is available.
     */
    [[nodiscard]] virtual bool has_exact_solution() const = 0;

    /**
     * @brief Returns the exact solution function object (if available).
     */
    virtual const Function<dim>& get_exact_solution() const = 0;

    /**
     * @brief Checks if the problem system matrix is symmetric.
     * @return true if symmetric (e.g., pure diffusion), false otherwise (e.g., with advection).
     */
    [[nodiscard]] virtual bool is_symmetric() const = 0;

    /**
     * @brief Returns a descriptive name of the problem.
     */
    [[nodiscard]] virtual std::string get_name() const = 0;
};

namespace Problems {

/**
 * @brief Manufactured exact solution: \f$ u(\mathbf{x}) = \prod \sin(\pi x_i) \f$.
 *
 * This function is used to verify the solver accuracy. It satisfies homogeneous Dirichlet
 * boundary conditions on the unit hypercube (except where we manually apply Neumann).
 */
template <int dim>
class ExactSolution : public Function<dim> {
public:
    /**
     * @brief Evaluates the exact solution at point p.
     * @param p The evaluation point.
     */
    double value(const Point<dim>& p, const unsigned int) const override {
        double val = 1.0;
        for (unsigned int d = 0; d < dim; ++d)
            val *= std::sin(PI * p[d]);
        return val;
    }

    /**
     * @brief Evaluates the gradient of the exact solution at point p.
     */
    Tensor<1, dim> gradient(const Point<dim>& p, const unsigned int) const override {
        Tensor<1, dim> grad;
        for (unsigned int d = 0; d < dim; ++d) {
            grad[d] = PI * std::cos(PI * p[d]);
            for (unsigned int other = 0; other < dim; ++other)
                if (d != other) grad[d] *= std::sin(PI * p[other]);
        }
        return grad;
    }
};

/**
 * @brief Analytic Neumann flux for the Right boundary (x=1).
 *
 * Calculates \f$ \nabla u \cdot \mathbf{n} \f$ at x=1, where \f$ \mathbf{n} = (1, 0, \dots)^T \f$.
 * Since \f$ \cos(\pi) = -1 \f$, the flux is negative.
 */
template <int dim>
class NeumannFluxRight : public Function<dim> {
public:
    double value(const Point<dim>& p, const unsigned int) const override {
        // Flux = du/dx at x=1.
        // u = sin(pi*x) * sin(pi*y)...
        // du/dx = pi * cos(pi*x) * sin(pi*y)...
        // At x=1, cos(pi*x) = -1.
        double val = -1.0 * PI;
        for (unsigned int d = 1; d < dim; ++d)
            val *= std::sin(PI * p[d]);
        return val;
    }
};

/**
 * @brief Concrete implementation of a steady-state Advection-Diffusion-Reaction problem.
 *
 * Solves: \f$ -\mu \Delta u + \mathbf{\beta} \cdot \nabla u + \gamma u = f \f$
 *
 * Domain: Unit Hypercube \f$ [0,1]^d \f$.
 * BCs: Neumann on Right face (ID 1), Homogeneous Dirichlet elsewhere.
 */
template <int dim>
class ADRProblem : public ProblemInterface<dim> {
public:
    ADRProblem()
        : zero_function(1), neumann_flux(), exact_solution() {}

    /** @brief Constant diffusion coefficient \f$ \mu = 1.0 \f$. */
    double diffusion_coefficient(const Point<dim>&) const override { return 1.0; }

    /** @brief Constant reaction coefficient \f$ \gamma = 0.1 \f$. */
    double reaction_coefficient(const Point<dim>&) const override { return 0.1; }

    /**
     * @brief Rotational advection field.
     * 2D: \f$ \beta = (-y, x)^T \f$
     * 3D: \f$ \beta = (-y, x, 0.1)^T \f$
     */
    Tensor<1, dim> advection_field(const Point<dim>& p) const override {
        Tensor<1, dim> beta;
        beta[0] = -p[1];
        beta[1] = p[0];
        if (dim == 3) beta[2] = 0.1;
        return beta;
    }

    /**
     * @brief Computes the source term \f$ f \f$ using the Method of Manufactured Solutions.
     *
     * \f$ f = -\mu \Delta u + \mathbf{\beta} \cdot \nabla u + \gamma u \f$
     *
     * where \f$ u \f$ is the ExactSolution.
     */
    double source_term(const Point<dim>& p) const override {
        const double u = exact_solution.value(p, 0);
        Tensor<1, dim> grad = exact_solution.gradient(p, 0);

        const double mu = diffusion_coefficient(p);
        const double gamma = reaction_coefficient(p);
        Tensor<1, dim> beta = advection_field(p);

        // Laplacian of u = product(sin(pi*x_i)) is -d * pi^2 * u
        const double lap = -1.0 * dim * PI * PI * u;

        return -mu * lap + beta * grad + gamma * u;
    }

    /**
     * @brief Defines Dirichlet boundaries.
     * All boundaries except Right (ID 1) are Dirichlet (0, 2, 3, 4, 5).
     */
    [[nodiscard]] std::set<types::boundary_id> get_dirichlet_ids() const override {
        std::set<types::boundary_id> ids;
        ids.insert(0); // Left
        ids.insert(2); // Bottom
        ids.insert(3); // Top
        if(dim == 3) { ids.insert(4); ids.insert(5); }
        return ids;
    }

    /**
     * @brief Defines Neumann boundaries.
     * Only the Right boundary (ID 1) is Neumann.
     */
    [[nodiscard]] std::set<types::boundary_id> get_neumann_ids() const override {
        return {1};
    }

    /** @brief Returns zero function for Dirichlet BCs. */
    const Function<dim>& get_dirichlet_function(types::boundary_id) const override {
        return zero_function;
    }

    /** @brief Returns analytic flux for Neumann BCs. */
    const Function<dim>& get_neumann_function(const types::boundary_id id) const override {
        if (id == 1) return neumann_flux;
        Assert(false, ExcMessage("Unknown Neumann ID requested"));
        return zero_function;
    }

    [[nodiscard]] bool has_exact_solution() const override { return true; }
    const Function<dim>& get_exact_solution() const override { return exact_solution; }
    
    /** @brief Problem is non-symmetric due to advection term. */
    [[nodiscard]] bool is_symmetric() const override { return false; }
    
    [[nodiscard]] std::string get_name() const override { return "Manufactured ADR (Dirichlet + Neumann)"; }

private:
    Functions::ZeroFunction<dim> zero_function;
    NeumannFluxRight<dim> neumann_flux;
    ExactSolution<dim> exact_solution;
};

} // namespace Problems
} // namespace HybridADRSolver

#endif

#endif // HYBRIDADRSOLVER_PROBLEM_DEFINITION_H