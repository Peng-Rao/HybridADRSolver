#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector.h>

#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
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

template <int dim> class DiffusionCoefficient : public Function<dim> {
public:
    double value(const Point<dim>& p,
                 const unsigned int component = 0) const override {
        (void)p;
        (void)component;
        return 1.0;
    }
};

template <int dim> class AdvectionField : public TensorFunction<1, dim> {
public:
    Tensor<1, dim> value(const Point<dim>& p) const override {
        (void)p;
        Tensor<1, dim> beta;
        beta[0] = 1.0;
        if (dim > 1)
            beta[1] = 0.0;
        if (dim > 2)
            beta[2] = 0.0;
        return beta;
    }
};

template <int dim> class ReactionCoefficient : public Function<dim> {
public:
    virtual double value(const Point<dim>& p,
                         const unsigned int component = 0) const override {
        (void)p;
        (void)component;
        return 1.0;
    }
};

template <int dim> class SourceTerm : public Function<dim> {
public:
    virtual double value(const Point<dim>& p,
                         const unsigned int component = 0) const override {
        (void)component;
        return 1.0;
    }
};

template <int dim> class DirichletBoundaryValues : public Function<dim> {
public:
    double value(const Point<dim>& p,
                 const unsigned int component) const override {
        (void)p;
        (void)component;
        return 0.0;
    }
};

template <int dim> class NeumannBoundaryValues : public Function<dim> {
public:
    double value(const Point<dim>& p,
                 const unsigned int component = 0) const override {
        (void)p;
        (void)component;
        return 0.0;
    }
};

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

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    PETScWrappers::MPI::SparseMatrix system_matrix;
    PETScWrappers::MPI::Vector locally_relevant_solution;
    PETScWrappers::MPI::Vector system_rhs;

    ConditionalOStream pcout;

    const types::boundary_id dirichlet_boundary_id = 0;
    const types::boundary_id neumann_boundary_id = 1;
};

template <int dim>
Solver<dim>::Solver(const unsigned int degree)
    : mpi_communicator(MPI_COMM_WORLD),
      triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                        Triangulation<dim>::smoothing_on_refinement |
                        Triangulation<dim>::smoothing_on_coarsening)),
      fe(degree), dof_handler(triangulation),
      pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)) {}

template <int dim> void Solver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();

    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::interpolate_boundary_values(dof_handler, dirichlet_boundary_id,
                                             DirichletBoundaryValues<dim>(),
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    SparsityTools::distribute_sparsity_pattern(
        dsp, dof_handler.locally_owned_dofs(), mpi_communicator,
        locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                         mpi_communicator);

    locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                                     mpi_communicator);

    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl;
    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;
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

    DiffusionCoefficient<dim> mu;
    AdvectionField<dim> beta;
    ReactionCoefficient<dim> gamma;
    SourceTerm<dim> f;
    NeumannBoundaryValues<dim> h;

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (cell->is_locally_owned() == false)
            continue;

        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        for (unsigned int q = 0; q < n_q_points; ++q) {
            const Point<dim>& x = fe_values.quadrature_point(q);
            double mu_val = mu.value(x);
            Tensor<1, dim> beta_val = beta.value(x);
            double gamma_val = gamma.value(x);
            double f_val = f.value(x);
            double JxW = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const double phi_i = fe_values.shape_value(i, q);
                const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const double phi_j = fe_values.shape_value(j, q);
                    const Tensor<1, dim> grad_phi_j =
                        fe_values.shape_grad(j, q);

                    // Diffusion + Advection + Reaction
                    cell_matrix(i, j) +=
                        (mu_val * grad_phi_j * grad_phi_i + // Diffusion
                         (beta_val * grad_phi_j) * phi_i +  // Advection
                         gamma_val * phi_j * phi_i) *
                        JxW; // Reaction
                }
                cell_rhs(i) += f_val * phi_i * JxW;
            }
        }

        // Neumann Boundary
        for (const auto& face : cell->face_iterators()) {
            if (face->at_boundary() &&
                face->boundary_id() == neumann_boundary_id) {
                fe_face_values.reinit(cell, face);
                for (unsigned int q = 0; q < n_face_q_points; ++q) {
                    double h_val = h.value(fe_face_values.quadrature_point(q));
                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        cell_rhs(i) += h_val *
                                       fe_face_values.shape_value(i, q) *
                                       fe_face_values.JxW(q);
                    }
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                               local_dof_indices, system_matrix,
                                               system_rhs);
    }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}

template <int dim> void Solver<dim>::solve() {
    PETScWrappers::MPI::Vector completely_distributed_solution(
        locally_owned_dofs, mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(),
                                 1e-12 * system_rhs.l2_norm());

    PETScWrappers::SolverGMRES solver(solver_control, mpi_communicator);

    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);

    solver.solve(system_matrix, completely_distributed_solution, system_rhs,
                 preconditioner);

    constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
}

template <int dim> void Solver<dim>::refine_grid() {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
        dof_handler, QGauss<dim - 1>(fe.degree + 1), {},
        locally_relevant_solution, estimated_error_per_cell, ComponentMask(),
        nullptr, 0, triangulation.locally_owned_subdomain());

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, estimated_error_per_cell, 0.3, 0.03);

    triangulation.execute_coarsening_and_refinement();
}

template <int dim>
void Solver<dim>::output_results(const unsigned int cycle) const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "solution");

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    const unsigned int n_ranks =
        Utilities::MPI::n_mpi_processes(mpi_communicator);
    const unsigned int my_rank =
        Utilities::MPI::this_mpi_process(mpi_communicator);

    const std::string filename = "solution-" + std::to_string(cycle) + "." +
                                 std::to_string(my_rank) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    if (my_rank == 0) {
        std::vector<std::string> filenames;
        filenames.reserve(n_ranks);
        for (unsigned int i = 0; i < n_ranks; ++i)
            filenames.push_back("solution-" + std::to_string(cycle) + "." +
                                std::to_string(i) + ".vtu");

        std::ofstream master_output("solution-" + std::to_string(cycle) +
                                    ".pvtu");
        data_out.write_pvtu_record(master_output, filenames);

        std::cout << "   Output written to solution-" << cycle << ".pvtu"
                  << std::endl;
    }
}

template <int dim> void Solver<dim>::run() {
    pcout << "Running on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)." << std::endl;

    constexpr unsigned int n_cycles = 6;

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle) {

        if (cycle == 0) {
            GridGenerator::hyper_cube(triangulation, 0.0, 1.0);

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

int main(int argc, char* argv[]) {
    try {
        using namespace GalerkinSolver;

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        Solver<2> problem_2d(1);
        problem_2d.run();

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