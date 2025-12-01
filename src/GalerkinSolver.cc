#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

namespace GalerkinSolver {
using namespace dealii;

// ============================================================
// Problem coefficients and data
// ============================================================
/**
 * Diffusion coefficient μ(x)
 */
template <int dim> class DiffusionCoefficient : public Function<dim> {
public:
    double value(const Point<dim>& p,
                 const unsigned int component = 0) const override {
        (void)p;
        (void)component;
        return 1.0; // Constant diffusion
    }
};

/**
 * Advection field β(x)
 */
template <int dim> class AdvectionField : public TensorFunction<1, dim> {
public:
    Tensor<1, dim> value(const Point<dim>& p) const override {
        (void)p;
        Tensor<1, dim> beta;
        beta[0] = 1.0; // Convection in x-direction
        if (dim > 1)
            beta[1] = 0.0;
        if (dim > 2)
            beta[2] = 0.0;
        return beta;
    }
};

/**
 * Reaction coefficient γ(x)
 */
template <int dim> class ReactionCoefficient : public Function<dim> {
public:
    virtual double value(const Point<dim>& p,
                         const unsigned int component = 0) const override {
        (void)p;
        (void)component;
        return 1.0; // Constant reaction
    }
};

/**
 * Source term f(x)
 */
template <int dim> class SourceTerm : public Function<dim> {
public:
    virtual double value(const Point<dim>& p,
                         const unsigned int component = 0) const override {
        (void)component;
        // Example: f = 1 for a simple test
        return 1.0;
    }
};

/**
 * Dirichlet boundary condition g(x)
 */
template <int dim> class DirichletBoundaryValues : public Function<dim> {
public:
    double value(const Point<dim>& p,
                 const unsigned int component) const override {
        (void)p;
        (void)component;
        return 0.0; // Homogeneous Dirichlet
    }
};

/**
 * Neumann boundary condition h(x) = ∇u·n
 */
template <int dim> class NeumannBoundaryValues : public Function<dim> {
public:
    double value(const Point<dim>& p,
                 const unsigned int component = 0) const override {
        (void)p;
        (void)component;
        return 0.0; // Homogeneous Neumann
    }
};

// ============================================================
// Main solver class
// ============================================================
template <int dim> class Solver {
public:
    explicit Solver(unsigned int degree = 1);
    void run();

private:
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(unsigned int cycle) const;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<> constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;

    // Boundary indicators
    // Γ_D: boundary_id = 0 (Dirichlet)
    // Γ_N: boundary_id = 1 (Neumann)
    const types::boundary_id dirichlet_boundary_id = 0;
    const types::boundary_id neumann_boundary_id = 1;
};

template <int dim>
Solver<dim>::Solver(const unsigned int degree)
    : fe(degree), dof_handler(triangulation) {}

template <int dim> void Solver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    // Setup constraints (Dirichlet BC)
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // Apply Dirichlet boundary conditions
    VectorTools::interpolate_boundary_values(dof_handler, dirichlet_boundary_id,
                                             DirichletBoundaryValues<dim>(),
                                             constraints);
    constraints.close();

    // Setup sparsity pattern
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim> void Solver<dim>::assemble_system() {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                         update_normal_vectors |
                                         update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Coefficient functions
    DiffusionCoefficient<dim> mu;
    AdvectionField<dim> beta;
    ReactionCoefficient<dim> gamma;
    SourceTerm<dim> f;
    NeumannBoundaryValues<dim> h;

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        // Get coefficient values at quadrature points
        std::vector<double> mu_values(n_q_points);
        std::vector<Tensor<1, dim>> beta_values(n_q_points);
        std::vector<double> gamma_values(n_q_points);
        std::vector<double> f_values(n_q_points);

        for (unsigned int q = 0; q < n_q_points; ++q) {
            const Point<dim>& x = fe_values.quadrature_point(q);
            mu_values[q] = mu.value(x);
            beta_values[q] = beta.value(x);
            gamma_values[q] = gamma.value(x);
            f_values[q] = f.value(x);
        }

        // Compute cell contribution
        for (unsigned int q = 0; q < n_q_points; ++q) {
            const double JxW = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const double phi_i = fe_values.shape_value(i, q);
                const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const double phi_j = fe_values.shape_value(j, q);
                    const Tensor<1, dim> grad_phi_j =
                        fe_values.shape_grad(j, q);

                    // Diffusion term: ∫ μ ∇φ_j · ∇φ_i dx
                    cell_matrix(i, j) +=
                        mu_values[q] * grad_phi_j * grad_phi_i * JxW;

                    // Convection term: ∫ (β · ∇φ_j) φ_i dx
                    // (using weak form after integration by parts)
                    cell_matrix(i, j) +=
                        (beta_values[q] * grad_phi_j) * phi_i * JxW;

                    // Reaction term: ∫ γ φ_j φ_i dx
                    cell_matrix(i, j) += gamma_values[q] * phi_j * phi_i * JxW;
                }

                // RHS: ∫ f φ_i dx
                cell_rhs(i) += f_values[q] * phi_i * JxW;
            }
        }

        // Neumann boundary contribution: ∫_Γ_N h φ_i ds
        for (const auto& face : cell->face_iterators()) {
            if (face->at_boundary() &&
                face->boundary_id() == neumann_boundary_id) {
                fe_face_values.reinit(cell, face);

                for (unsigned int q = 0; q < n_face_q_points; ++q) {
                    const double h_value =
                        h.value(fe_face_values.quadrature_point(q));

                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        cell_rhs(i) += h_value *
                                       fe_face_values.shape_value(i, q) *
                                       fe_face_values.JxW(q);
                    }
                }
            }
        }

        // Transfer to global system
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                               local_dof_indices, system_matrix,
                                               system_rhs);
    }
}

template <int dim> void Solver<dim>::solve() {
    // Use GMRES for non-symmetric system
    SolverControl solver_control(1000, 1e-12);
    SolverGMRES<Vector<double>> solver(solver_control);

    // ILU preconditioner
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    std::cout << "   " << solver_control.last_step()
              << " GMRES iterations needed to obtain convergence." << std::endl;
}

template <int dim> void Solver<dim>::refine_grid() {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1), {},
                                       solution, estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, estimated_error_per_cell, 0.3, 0.03);

    triangulation.execute_coarsening_and_refinement();
}

template <int dim>
void Solver<dim>::output_results(const unsigned int cycle) const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();

    const std::string filename = "solution-" + std::to_string(cycle) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);

    std::cout << "   Output written to " << filename << std::endl;
}

template <int dim> void Solver<dim>::run() {
    constexpr unsigned int n_cycles = 5;

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle) {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0) {
            // Create initial mesh
            GridGenerator::hyper_cube(triangulation, 0.0, 1.0);

            // Set boundary indicators:
            // Left boundary (x=0): Dirichlet (id=0)
            // Other boundaries: Neumann (id=1)
            for (const auto& cell : triangulation.cell_iterators())
                for (const auto& face : cell->face_iterators())
                    if (face->at_boundary()) {
                        const Point<dim> center = face->center();
                        if (std::abs(center[0]) < 1e-12)
                            face->set_boundary_id(dirichlet_boundary_id);
                        else
                            face->set_boundary_id(neumann_boundary_id);
                    }

            triangulation.refine_global(3);
        } else {
            refine_grid();
        }

        setup_system();
        assemble_system();
        solve();
        output_results(cycle);
    }
}

} // namespace GalerkinSolver

int main() {
    try {
        using namespace GalerkinSolver;

        std::cout << "Solving 2D Convection-Diffusion-Reaction problem"
                  << std::endl;
        std::cout << "================================================"
                  << std::endl;

        Solver<2> problem_2d(1); // Linear elements
        problem_2d.run();

        std::cout << std::endl;
    } catch (std::exception& exc) {
        std::cerr << std::endl
                  << "--------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl;
        return 1;
    } catch (...) {
        std::cerr << std::endl
                  << "--------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl;
        return 1;
    }

    return 0;
}
