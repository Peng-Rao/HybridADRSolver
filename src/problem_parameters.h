//
// Created by PENG RAO on 03/12/25.
//

#ifndef HYBRIDADRSOLVER_PROBLEM_PARAMETERS_H
#define HYBRIDADRSOLVER_PROBLEM_PARAMETERS_H

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

namespace parameters {
using namespace dealii;

// Problem dimension
constexpr unsigned int dim = 3;

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
template <int dim> class SourceTerm : public Function<dim> {
public:
    explicit SourceTerm() : Function<dim>() {}

    double value(const Point<dim>& p,
                 const unsigned int component) const override {
        (void)component;
        // Gaussian source term centered at origin
        const double r_squared = p.norm_square();
        return std::exp(-10.0 * r_squared);
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

} // namespace parameters

#endif // HYBRIDADRSOLVER_PROBLEM_PARAMETERS_H
