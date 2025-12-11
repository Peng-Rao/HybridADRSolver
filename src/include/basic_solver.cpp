#include "basic_solver.h"
#include "problem_parameters.h"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#ifdef USE_PETSC_MPI
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/sparsity_tools.h>
#endif

#include <fstream>

namespace BasicSolver {
using namespace parameters;

template <int dim>
Solver<dim>::Solver(const unsigned int degree)
    : mpi_communicator(MPI_COMM_WORLD),
#ifdef USE_PETSC_MPI
      triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                        Triangulation<dim>::smoothing_on_refinement |
                        Triangulation<dim>::smoothing_on_coarsening)),
#else
      triangulation(Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening),
#endif
      fe(degree), dof_handler(triangulation),
      pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)) {
}

template <int dim> void Solver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
        dof_handler, dirichlet_boundary_id,
        parameters::DirichletBoundaryValues<dim>(), constraints);
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);

#ifdef USE_PETSC_MPI
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(
        dsp, dof_handler.locally_owned_dofs(), mpi_communicator,
        locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                         mpi_communicator);
    locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
#else
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    locally_relevant_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
#endif

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl;
    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;
}

template <int dim> void Solver<dim>::assemble_system() {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_hessians |
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
    NeumannBoundaryValues<dim> h_boundary;

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (cell->is_locally_owned() == false)
            continue;

        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        // Retrieve cell diameter for stabilization parameter calculation
        const double h_K = cell->diameter();

        for (unsigned int q = 0; q < n_q_points; ++q) {
            const Point<dim>& x = fe_values.quadrature_point(q);
            const double mu_val = mu.value(x, 0);
            const Tensor<1, dim> beta_val = beta.value(x);
            const double gamma_val = gamma.value(x, 0);
            const double f_val = f.value(x, 0);
            const double JxW = fe_values.JxW(q);

            // Calculate SUPG stabilization parameter (tau)
            // Using a standard asymptotic formula for
            // advection-diffusion-reaction
            const double beta_norm = beta_val.norm();
            double tau = 0.0;

            if (beta_norm > 1e-12) {
                const double num_sq = std::pow(2.0 * beta_norm / h_K, 2);
                const double diff_sq = std::pow(4.0 * mu_val / (h_K * h_K), 2);
                const double react_sq = std::pow(gamma_val, 2);
                tau = 1.0 / std::sqrt(num_sq + diff_sq + react_sq);
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const double phi_i = fe_values.shape_value(i, q);
                const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

                // SUPG Test Function term: tau * (beta * grad(v))
                const double supg_test_i = tau * (beta_val * grad_phi_i);

                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const double phi_j = fe_values.shape_value(j, q);
                    const Tensor<1, dim> grad_phi_j =
                        fe_values.shape_grad(j, q);

                    // Standard Galerkin Contribution
                    double cell_contribution =
                        (mu_val * grad_phi_j * grad_phi_i +
                         (beta_val * grad_phi_j) * phi_i +
                         gamma_val * phi_j * phi_i);

                    // SUPG Stabilization Contribution
                    // Strong Residual: -mu * laplacian(u) + beta * grad(u) +
                    // gamma * u Note: trace(hessian) gives the Laplacian
                    const double laplacian_phi_j =
                        trace(fe_values.shape_hessian(j, q));
                    const double strong_residual_j = -mu_val * laplacian_phi_j +
                                                     (beta_val * grad_phi_j) +
                                                     gamma_val * phi_j;

                    cell_contribution += strong_residual_j * supg_test_i;

                    cell_matrix(i, j) += cell_contribution * JxW;
                }

                // Standard RHS + SUPG RHS
                // RHS term: (f, v + tau * beta * grad(v))
                cell_rhs(i) += (f_val * phi_i + f_val * supg_test_i) * JxW;
            }
        }

        for (const auto& face : cell->face_iterators()) {
            if (face->at_boundary() &&
                face->boundary_id() == neumann_boundary_id) {
                fe_face_values.reinit(cell, face);
                for (unsigned int q = 0; q < n_face_q_points; ++q) {
                    double h_val =
                        h_boundary.value(fe_face_values.quadrature_point(q), 0);
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

#ifdef USE_PETSC_MPI
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
#endif
}

template <int dim> void Solver<dim>::solve() {
#ifdef USE_PETSC_MPI
    PETScWrappers::MPI::Vector completely_distributed_solution(
        locally_owned_dofs, mpi_communicator);
    SolverControl solver_control(dof_handler.n_dofs(),
                                 1e-12 * system_rhs.l2_norm());
    PETScWrappers::SolverGMRES solver(solver_control);
    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
    solver.solve(system_matrix, completely_distributed_solution, system_rhs,
                 preconditioner);
    constraints.distribute(completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
    (void)(pcout << "   Solved in " << solver_control.last_step()
                 << " iterations." << std::endl);
#else
    SolverControl solver_control(1000, 1e-12);
    SolverGMRES<VectorType> solver(solver_control);
    PreconditionSSOR<MatrixType> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    solver.solve(system_matrix, locally_relevant_solution, system_rhs,
                 preconditioner);
    constraints.distribute(locally_relevant_solution);
    (void)(pcout << "   Solved in " << solver_control.last_step()
                 << " iterations." << std::endl);
#endif
}

template <int dim> void Solver<dim>::refine_grid() {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
        dof_handler, QGauss<dim - 1>(fe.degree + 1), {},
        locally_relevant_solution, estimated_error_per_cell, ComponentMask(),
        nullptr, 0, triangulation.locally_owned_subdomain());

#ifdef USE_PETSC_MPI
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, estimated_error_per_cell, 0.3, 0.03);
#else
    GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, estimated_error_per_cell, 0.3, 0.03);
#endif
    triangulation.execute_coarsening_and_refinement();
}

template <int dim> void Solver<dim>::output_results(unsigned int cycle) const {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "solution");

#ifdef USE_PETSC_MPI
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
#endif

    data_out.build_patches();
    const std::string filename_base = "solution-" + std::to_string(cycle);

#ifdef USE_PETSC_MPI
    const unsigned int n_ranks =
        Utilities::MPI::n_mpi_processes(mpi_communicator);
    const unsigned int my_rank =
        Utilities::MPI::this_mpi_process(mpi_communicator);
    const std::string filename_vtu =
        filename_base + "." + std::to_string(my_rank) + ".vtu";
    std::ofstream output(filename_vtu);
    data_out.write_vtu(output);
    if (my_rank == 0) {
        std::vector<std::string> filenames;
        filenames.reserve(n_ranks);
        for (unsigned int i = 0; i < n_ranks; ++i)
            filenames.push_back(filename_base + "." + std::to_string(i) +
                                ".vtu");
        std::ofstream master_output(filename_base + ".pvtu");
        data_out.write_pvtu_record(master_output, filenames);
        (void)(pcout << "   Output written to " << filename_base << ".pvtu"
                     << std::endl);
    }
#else
    const std::string filename_vtk = filename_base + ".vtk";
    std::ofstream output(filename_vtk);
    data_out.write_vtk(output);
    (void)(pcout << "   Output written to " << filename_vtk << std::endl);
#endif
}

template <int dim> void Solver<dim>::run() {
    pcout << "Running in "
#ifdef USE_PETSC_MPI
          << "PARALLEL (MPI/PETSc)"
#else
          << "SEQUENTIAL"
#endif
          << " mode on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)." << std::endl;

    constexpr unsigned int n_cycles = 6;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle) {
        (void)(pcout << "Cycle " << cycle << ":" << std::endl);
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
template class Solver<2>;

} // namespace BasicSolver