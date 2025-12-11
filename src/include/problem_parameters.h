#ifndef HYBRIDADRSOLVER_PROBLEM_PARAMETERS_H
#define HYBRIDADRSOLVER_PROBLEM_PARAMETERS_H

#include <cmath>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor_function.h>

namespace parameters {
using namespace dealii;
using numbers::PI;

// Problem dimension
constexpr unsigned int dim = 3;

// Pick a smooth Exact Solution that is zero on unit cube boundaries
// u(x,y,z) = sin(pi*x) * sin(pi*y) * sin(pi*z)
template <int dim> class ExactSolution : public Function<dim> {
public:
    ExactSolution() : Function<dim>() {}

    double value(const Point<dim>& p, const unsigned int) const override {
        double val = 1.0;
        for (unsigned int d = 0; d < dim; ++d)
            val *= std::sin(PI * p[d]);
        return val;
    }

    Tensor<1, dim> gradient(const Point<dim>& p,
                            const unsigned int) const override {
        Tensor<1, dim> grad;
        for (unsigned int d = 0; d < dim; ++d) {
            grad[d] = PI * std::cos(PI * p[d]);
            for (unsigned int other_d = 0; other_d < dim; ++other_d) {
                if (d != other_d)
                    grad[d] *= std::sin(PI * p[other_d]);
            }
        }
        return grad;
    }
};

/**
 * Diffusion coefficient $\mu(x)$
 * For simplicity, we use a constant coefficient, but this can be spatially
 * varying
 */
template <int dim> class DiffusionCoefficient : public Function<dim> {
public:
    explicit DiffusionCoefficient(const double value = 1.0)
        : Function<dim>(), coefficient_value(value) {}

    double value(const Point<dim>& p,
                 const unsigned int component) const override {
        (void)p;
        (void)component;
        return coefficient_value;
    }

private:
    double coefficient_value;
};

// /**
//  * Advection velocity field β(x)
//  * This represents the advection/transport direction
//  */
template <int dim> class AdvectionField final : public TensorFunction<1, dim> {
public:
    AdvectionField() : TensorFunction<1, dim>() {}

    Tensor<1, dim> value(const Point<dim>& p) const override {
        Tensor<1, dim> beta;
        beta[0] = -p[1];
        beta[1] = p[0];
        if constexpr (dim == 3)
            beta[2] = 0.1;
        return beta;
    }
};

/**
 * Reaction coefficient γ(x)
 */
template <int dim> class ReactionCoefficient : public Function<dim> {
public:
    explicit ReactionCoefficient(const double value = 0.1)
        : Function<dim>(), coefficient_value(value) {}

    double value(const Point<dim>& p,
                 const unsigned int component) const override {
        (void)p;
        (void)component;
        return coefficient_value;
    }

private:
    double coefficient_value;
};

/**
 * Source term f(x)
 */
/**
 * Right-hand side function f(x)
 */
// We compute f = -mu*Laplacian(u) + beta*grad(u) + gamma*u
// using the ExactSolution u defined above.
template <int dim> class SourceTerm : public Function<dim> {
public:
    SourceTerm() : Function<dim>() {}

    double value(const Point<dim>& p, const unsigned int) const override {
        double u_val = 1.0;
        for (unsigned int d = 0; d < dim; ++d)
            u_val *= std::sin(PI * p[d]);

        Tensor<1, dim> grad_u;
        for (unsigned int d = 0; d < dim; ++d) {
            grad_u[d] = PI * std::cos(PI * p[d]);
            for (unsigned int other_d = 0; other_d < dim; ++other_d)
                if (d != other_d)
                    grad_u[d] *= std::sin(PI * p[other_d]);
        }

        const double laplacian_u = -1.0 * dim * PI * PI * u_val;

        constexpr double mu = 1.0;    //  DiffusionCoefficient
        constexpr double gamma = 0.1; //  ReactionCoefficient

        Tensor<1, dim> beta;
        beta[0] = -p[1];
        beta[1] = p[0];
        if (dim == 3)
            beta[2] = 0.1;

        // Combine into Residual ---
        return -mu * laplacian_u + beta * grad_u + gamma * u_val;
    }
};

/**
 * Dirichlet boundary condition g(x)
 */
template <int dim> class DirichletBoundaryValues : public Function<dim> {
public:
    DirichletBoundaryValues() : Function<dim>() {}

    double value(const Point<dim>& p,
                 const unsigned int component) const override {
        (void)p;
        (void)component;
        return 0.0; // Homogeneous Dirichlet BC
    }
};

/**
 * Neumann boundary condition $h(x)=\gradient u \dot n$
 */
template <int dim> class NeumannBoundaryValues : public Function<dim> {
public:
    NeumannBoundaryValues() : Function<dim>() {}

    double value(const Point<dim>& p,
                 const unsigned int component) const override {
        (void)p;
        (void)component;
        return 0.0; // Homogeneous Neumann BC
    }
};

/**
 * Runtime parameters structure
 */
struct RuntimeParameters {
    unsigned int fe_degree = 2;
    unsigned int n_global_refines = 4;
    unsigned int n_cycles = 1;
    double diffusion_coefficient = 1.0;
    double reaction_coefficient = 0.1;
    std::string output_filename = "solution";
    bool output_vtu = true;

    static void declare_parameters(ParameterHandler& prm) {
        prm.declare_entry("Finite element degree", "2", Patterns::Integer(1),
                          "Polynomial degree of finite elements");
        prm.declare_entry("Number of global refinements", "4",
                          Patterns::Integer(0),
                          "Number of global mesh refinements");
        prm.declare_entry("Number of cycles", "1", Patterns::Integer(1),
                          "Number of refinement cycles");
        prm.declare_entry("Diffusion coefficient", "1.0", Patterns::Double(0),
                          "Value of diffusion coefficient mu");
        prm.declare_entry("Reaction coefficient", "0.1", Patterns::Double(0),
                          "Value of reaction coefficient gamma");
        prm.declare_entry("Output filename", "solution", Patterns::Anything(),
                          "Base name for output files");
        prm.declare_entry("Output VTU", "true", Patterns::Bool(),
                          "Whether to output VTU files");
    };

    void parse_parameters(const ParameterHandler& prm) {
        fe_degree = prm.get_integer("Finite element degree");
        n_global_refines = prm.get_integer("Number of global refinements");
        n_cycles = prm.get_integer("Number of cycles");
        diffusion_coefficient = prm.get_double("Diffusion coefficient");
        reaction_coefficient = prm.get_double("Reaction coefficient");
        output_filename = prm.get("Output filename");
        output_vtu = prm.get_bool("Output VTU");
    }
};
} // namespace parameters

#endif // HYBRIDADRSOLVER_PROBLEM_PARAMETERS_H
