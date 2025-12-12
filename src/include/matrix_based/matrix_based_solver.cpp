/**
 * @file matrix_based_solver.cpp
 * @brief Implementation of Matrix-based solver
 */

#include "matrix_based/matrix_based_solver.h"
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

namespace HybridADRSolver {
using namespace dealii;

// ==========================================================================
// Helper Struct Definitions
// ==========================================================================

template <int dim> struct ScratchData {
    ScratchData(const Mapping<dim>& mapping, const FiniteElement<dim>& fe,
                const Quadrature<dim>& quadrature,
                const Quadrature<dim - 1>& face_quadrature,
                const UpdateFlags update_flags,
                const UpdateFlags face_update_flags,
                const ProblemInterface<dim>& problem)
        : fe_values(mapping, fe, quadrature, update_flags),
          fe_face_values(mapping, fe, face_quadrature, face_update_flags),
          problem(problem) {}

    ScratchData(const ScratchData& scratch)
        : fe_values(scratch.fe_values.get_mapping(), scratch.fe_values.get_fe(),
                    scratch.fe_values.get_quadrature(),
                    scratch.fe_values.get_update_flags()),
          fe_face_values(scratch.fe_face_values.get_mapping(),
                         scratch.fe_face_values.get_fe(),
                         scratch.fe_face_values.get_quadrature(),
                         scratch.fe_face_values.get_update_flags()),
          problem(scratch.problem) {}

    FEValues<dim> fe_values;
    FEFaceValues<dim> fe_face_values;
    const ProblemInterface<dim>& problem;
};

struct CopyData {
    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
};

// ==========================================================================
// Class Implementation
// ==========================================================================

template <int dim>
MatrixBasedSolver<dim>::MatrixBasedSolver(const ProblemInterface<dim>& problem,
                                          const unsigned int degree,
                                          MPI_Comm comm,
                                          const SolverParameters& params)
    : ParallelSolverBase<dim>(comm, params), problem(problem),
      fe_degree(degree) {
    this->fe = std::make_unique<FE_Q<dim>>(degree);
    this->mapping = std::make_unique<MappingQ<dim>>(degree);
}

template <int dim> void MatrixBasedSolver<dim>::setup_dofs() {
    this->dof_handler.distribute_dofs(*this->fe);
    DoFRenumbering::Cuthill_McKee(this->dof_handler);

    this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    this->locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    this->constraints.clear();
    this->constraints.reinit(this->locally_owned_dofs,
                             this->locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            this->constraints);

    for (const auto& boundary_id : problem.get_dirichlet_boundary_ids()) {
        VectorTools::interpolate_boundary_values(this->dof_handler, boundary_id,
                                                 *problem.get_dirichlet_bc(),
                                                 this->constraints);
    }
    this->constraints.close();

    DynamicSparsityPattern dsp(this->locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(this->dof_handler, dsp, this->constraints,
                                    false);
    SparsityTools::distribute_sparsity_pattern(dsp, this->locally_owned_dofs,
                                               this->mpi_communicator,
                                               this->locally_relevant_dofs);

    system_matrix.reinit(this->locally_owned_dofs, this->locally_owned_dofs,
                         dsp, this->mpi_communicator);

    system_rhs.reinit(this->locally_owned_dofs, this->mpi_communicator);
    solution.reinit(this->locally_owned_dofs, this->locally_relevant_dofs,
                    this->mpi_communicator);

    this->timing_results.n_dofs = this->dof_handler.n_dofs();
    this->timing_results.memory_mb = (system_matrix.memory_consumption() +
                                      system_rhs.memory_consumption() * 2) /
                                     1024.0 / 1024.0;

    if (this->parameters.verbose) {
        this->pcout << "   DoFs: " << this->dof_handler.n_dofs() << std::endl;
        this->pcout << "   Memory: " << this->timing_results.memory_mb << " MB"
                    << std::endl;
    }
}

template <int dim>
void MatrixBasedSolver<dim>::local_assemble_cell(
    const typename DoFHandler<dim>::active_cell_iterator& cell,
    ScratchData<dim>& scratch, CopyData& copy_data) {
    // Safety check (though iterator filter handles this, it's good practice)
    if (!cell->is_locally_owned())
        return;

    const unsigned int dofs_per_cell =
        scratch.fe_values.get_fe().n_dofs_per_cell();
    const unsigned int n_q_points = scratch.fe_values.get_quadrature().size();

    copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    copy_data.cell_rhs.reinit(dofs_per_cell);
    copy_data.local_dof_indices.resize(dofs_per_cell);

    scratch.fe_values.reinit(cell);
    const auto& q_points = scratch.fe_values.get_quadrature_points();

    for (unsigned int q = 0; q < n_q_points; ++q) {
        const double mu = scratch.problem.diffusion(q_points[q]);
        const Tensor<1, dim> beta = scratch.problem.advection(q_points[q]);
        const double gamma = scratch.problem.reaction(q_points[q]);
        const double f = scratch.problem.source(q_points[q]);
        const double JxW = scratch.fe_values.JxW(q);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            const double phi_i = scratch.fe_values.shape_value(i, q);
            const Tensor<1, dim> grad_phi_i =
                scratch.fe_values.shape_grad(i, q);

            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const double phi_j = scratch.fe_values.shape_value(j, q);
                const Tensor<1, dim> grad_phi_j =
                    scratch.fe_values.shape_grad(j, q);

                copy_data.cell_matrix(i, j) +=
                    mu * (grad_phi_i * grad_phi_j) * JxW;
                copy_data.cell_matrix(i, j) += beta * grad_phi_j * phi_i * JxW;
                copy_data.cell_matrix(i, j) += gamma * phi_i * phi_j * JxW;
            }
            copy_data.cell_rhs(i) += f * phi_i * JxW;
        }
    }

    for (const auto& face : cell->face_iterators()) {
        if (face->at_boundary()) {
            if (const types::boundary_id bid = face->boundary_id();
                problem.get_neumann_boundary_ids().count(bid) > 0) {
                scratch.fe_face_values.reinit(cell, face);
                const auto& face_q_points =
                    scratch.fe_face_values.get_quadrature_points();
                const unsigned int n_face_q = face_q_points.size();

                for (unsigned int q = 0; q < n_face_q; ++q) {
                    const double h =
                        problem.get_neumann_bc()->value(face_q_points[q]);
                    const double JxW = scratch.fe_face_values.JxW(q);

                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        copy_data.cell_rhs(i) +=
                            h * scratch.fe_face_values.shape_value(i, q) * JxW;
                    }
                }
            }
        }
    }

    cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void MatrixBasedSolver<dim>::copy_local_to_global(const CopyData& copy_data) {
    // If the data is empty (e.g. from a filtered cell, though WorkStream
    // prevents this)
    if (copy_data.local_dof_indices.empty())
        return;

    this->constraints.distribute_local_to_global(
        copy_data.cell_matrix, copy_data.cell_rhs, copy_data.local_dof_indices,
        system_matrix, system_rhs);
}

template <int dim> void MatrixBasedSolver<dim>::assemble_system() {
    const QGauss<dim> quadrature(fe_degree + 1);
    const QGauss<dim - 1> face_quadrature(fe_degree + 1);

    const UpdateFlags update_flags = update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values;
    const UpdateFlags face_update_flags =
        update_values | update_quadrature_points | update_JxW_values;

    WorkStream::run(this->dof_handler.begin_active(), this->dof_handler.end(),
                    *this, &MatrixBasedSolver<dim>::local_assemble_cell,
                    &MatrixBasedSolver<dim>::copy_local_to_global,
                    ScratchData<dim>(*this->mapping, *this->fe, quadrature,
                                     face_quadrature, update_flags,
                                     face_update_flags, problem),
                    CopyData());

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}

template <int dim> void MatrixBasedSolver<dim>::solve() {
    LADistributed::MPI::Vector distributed_solution(this->locally_owned_dofs,
                                                    this->mpi_communicator);

    SolverControl solver_control(this->parameters.max_iterations,
                                 this->parameters.tolerance *
                                     system_rhs.l2_norm());

    if (problem.is_symmetric()) {
        LADistributed::SolverCG solver(solver_control);
        LADistributed::MPI::PreconditionAMG preconditioner;
        preconditioner.initialize(system_matrix);
        solver.solve(system_matrix, distributed_solution, system_rhs,
                     preconditioner);
    } else {
        LADistributed::SolverGMRES solver(solver_control);
        LADistributed::MPI::PreconditionJacobi preconditioner;
        preconditioner.initialize(system_matrix);
        solver.solve(system_matrix, distributed_solution, system_rhs,
                     preconditioner);
    }

    this->timing_results.n_iterations = solver_control.last_step();

    if (this->parameters.verbose) {
        this->pcout << "   Converged in " << solver_control.last_step()
                    << " iterations." << std::endl;
    }

    this->constraints.distribute(distributed_solution);
    solution = distributed_solution;
}

template <int dim>
void MatrixBasedSolver<dim>::output_results(const unsigned int cycle) const {
    if (!this->parameters.output_solution)
        return;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler);
    data_out.add_data_vector(solution, "solution");

    Vector<float> subdomain(this->triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = this->triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(*this->mapping);
    data_out.write_vtu_with_pvtu_record("./",
                                        this->parameters.output_prefix + "_mb",
                                        cycle, this->mpi_communicator, 2);
}

template <int dim>
void MatrixBasedSolver<dim>::run(const unsigned int n_refinements) {
    if (this->parameters.verbose) {
        this->pcout << "\n=== " << get_name() << " ===" << std::endl;
        this->pcout << "   Problem: " << problem.get_name() << std::endl;
        this->pcout << "   FE degree: " << fe_degree << std::endl;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    this->setup_grid(n_refinements);

    const auto t1 = std::chrono::high_resolution_clock::now();
    setup_dofs();
    const auto t2 = std::chrono::high_resolution_clock::now();
    this->timing_results.setup_time =
        std::chrono::duration<double>(t2 - t1).count();

    const auto t3 = std::chrono::high_resolution_clock::now();
    assemble_system();
    const auto t4 = std::chrono::high_resolution_clock::now();
    this->timing_results.assembly_time =
        std::chrono::duration<double>(t4 - t3).count();

    // Benchmark single operator application
    {
        LADistributed::MPI::Vector tmp(this->locally_owned_dofs,
                                       this->mpi_communicator);
        tmp = system_rhs;
        const auto ts = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i)
            system_matrix.vmult(system_rhs, tmp);
        const auto te = std::chrono::high_resolution_clock::now();
        this->timing_results.operator_apply_time =
            std::chrono::duration<double>(te - ts).count() / 10.0;
        system_rhs = 0;
        assemble_system(); // Re-assemble RHS
    }

    const auto t5 = std::chrono::high_resolution_clock::now();
    solve();
    const auto t6 = std::chrono::high_resolution_clock::now();
    this->timing_results.solve_time =
        std::chrono::duration<double>(t6 - t5).count();

    this->timing_results.total_time =
        std::chrono::duration<double>(t6 - t0).count();

    output_results(0);

    if (this->parameters.verbose) {
        this->pcout << "\n   Timing Summary:" << std::endl;
        this->pcout << "     Setup:     " << this->timing_results.setup_time
                    << " s" << std::endl;
        this->pcout << "     Assembly:  " << this->timing_results.assembly_time
                    << " s" << std::endl;
        this->pcout << "     Solve:     " << this->timing_results.solve_time
                    << " s" << std::endl;
        this->pcout << "     Total:     " << this->timing_results.total_time
                    << " s" << std::endl;
    }
}
// ==========================================================================
// Explicit Instantiation
// ==========================================================================
template class MatrixBasedSolver<2>;
template class MatrixBasedSolver<3>;

} // namespace HybridADRSolver