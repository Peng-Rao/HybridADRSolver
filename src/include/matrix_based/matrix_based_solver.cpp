/**
 *  @file matrix_based_solver.cpp
 * @brief Implementation of the Matrix-based Finite Element solver (Optimized).
 *
 * This file implements the `MatrixBasedSolver` class. It uses the deal.II
 * `WorkStream` functionality for multithreaded assembly and PETSc wrappers for
 * distributed linear algebra.
 *
 */

#include "matrix_based/matrix_based_solver.h"

#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <petscksp.h>
#include <petscpc.h>

namespace HybridADRSolver {

using namespace dealii;

/**
 * @brief Scratch data for the `WorkStream` assembly (Optimized).
 */
template <int dim> struct ScratchData {
    ScratchData(const FiniteElement<dim>& fe, const Quadrature<dim>& q,
                const Quadrature<dim - 1>& fq)
        : fe_values(fe, q,
                    update_values | update_gradients |
                        update_quadrature_points | update_JxW_values),
          fe_face_values(fe, fq,
                         update_values | update_quadrature_points |
                             update_JxW_values),
          n_dofs(fe.n_dofs_per_cell()), n_q_points(q.size()),
          n_face_q_points(fq.size()), phi(n_dofs), grad_phi(n_dofs),
          face_phi(n_dofs), mu_values(n_q_points), beta_values(n_q_points),
          gamma_values(n_q_points), f_values(n_q_points),
          JxW_values(n_q_points) {}

    ScratchData(const ScratchData& sd)
        : fe_values(sd.fe_values.get_fe(), sd.fe_values.get_quadrature(),
                    sd.fe_values.get_update_flags()),
          fe_face_values(sd.fe_face_values.get_fe(),
                         sd.fe_face_values.get_quadrature(),
                         sd.fe_face_values.get_update_flags()),
          n_dofs(sd.n_dofs), n_q_points(sd.n_q_points),
          n_face_q_points(sd.n_face_q_points), phi(sd.n_dofs),
          grad_phi(sd.n_dofs), face_phi(sd.n_dofs), mu_values(sd.n_q_points),
          beta_values(sd.n_q_points), gamma_values(sd.n_q_points),
          f_values(sd.n_q_points), JxW_values(sd.n_q_points) {}

    FEValues<dim> fe_values;
    FEFaceValues<dim> fe_face_values;

    const unsigned int n_dofs;
    const unsigned int n_q_points;
    const unsigned int n_face_q_points;

    std::vector<double> phi;
    std::vector<Tensor<1, dim>> grad_phi;
    std::vector<double> face_phi;

    std::vector<double> mu_values;
    std::vector<Tensor<1, dim>> beta_values;
    std::vector<double> gamma_values;
    std::vector<double> f_values;
    std::vector<double> JxW_values;
};

struct CopyData {
    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
};

template <int dim>
MatrixBasedSolver<dim>::MatrixBasedSolver(const ProblemInterface<dim>& prob,
                                          unsigned int degree, MPI_Comm comm,
                                          const SolverParameters& params)
    : ParallelSolverBase<dim>(comm, params), problem(prob), fe_degree(degree) {
    this->fe = std::make_unique<FE_Q<dim>>(degree);
    this->mapping = std::make_unique<MappingQ<dim>>(degree);
}

template <int dim> double MatrixBasedSolver<dim>::compute_memory_usage() const {
    double local_memory = 0.0;
    local_memory += system_matrix.memory_consumption();
    local_memory += solution.memory_consumption();
    local_memory += system_rhs.memory_consumption();
    local_memory += this->constraints.memory_consumption();
    const double global_memory =
        Utilities::MPI::sum(local_memory, this->mpi_communicator);
    return global_memory / (1024.0 * 1024.0);
}

template <int dim> void MatrixBasedSolver<dim>::setup_dofs() {
    this->dof_handler.distribute_dofs(*this->fe);
    this->timing_results.n_dofs = this->dof_handler.n_dofs();

    if (this->parameters.verbose) {
        this->pcout << "   Number of DoFs: " << this->dof_handler.n_dofs()
                    << std::endl;
    }

    DoFRenumbering::Cuthill_McKee(this->dof_handler);

    this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    this->locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    this->constraints.clear();
    this->constraints.reinit(this->locally_owned_dofs,
                             this->locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            this->constraints);

    for (const auto& id : problem.get_dirichlet_ids()) {
        VectorTools::interpolate_boundary_values(
            this->dof_handler, id, problem.get_dirichlet_function(id),
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
}

template <int dim> void MatrixBasedSolver<dim>::assemble_system() {
    auto worker =
        [&](const typename DoFHandler<dim>::active_cell_iterator& cell,
            ScratchData<dim>& scratch, CopyData& copy) {
            const unsigned int n_dofs = scratch.n_dofs;
            const unsigned int n_q_points = scratch.n_q_points;

            copy.cell_matrix.reinit(n_dofs, n_dofs);
            copy.cell_rhs.reinit(n_dofs);
            copy.local_dof_indices.resize(n_dofs);

            scratch.fe_values.reinit(cell);

            const auto& q_points = scratch.fe_values.get_quadrature_points();

            // Pre-evaluate all coefficients for this cell
            for (unsigned int q = 0; q < n_q_points; ++q) {
                scratch.mu_values[q] =
                    problem.diffusion_coefficient(q_points[q]);
                scratch.beta_values[q] = problem.advection_field(q_points[q]);
                scratch.gamma_values[q] =
                    problem.reaction_coefficient(q_points[q]);
                scratch.f_values[q] = problem.source_term(q_points[q]);
                scratch.JxW_values[q] = scratch.fe_values.JxW(q);
            }

            // Cell Integration Loop
            for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int k = 0; k < n_dofs; ++k) {
                    scratch.phi[k] = scratch.fe_values.shape_value(k, q);
                    scratch.grad_phi[k] = scratch.fe_values.shape_grad(k, q);
                }

                const double mu = scratch.mu_values[q];
                const auto& beta = scratch.beta_values[q];
                const double gamma = scratch.gamma_values[q];
                const double f = scratch.f_values[q];
                const double JxW = scratch.JxW_values[q];

                for (unsigned int i = 0; i < n_dofs; ++i) {
                    const double phi_i = scratch.phi[i];
                    const auto& grad_i = scratch.grad_phi[i];

                    for (unsigned int j = 0; j < n_dofs; ++j) {
                        const double phi_j = scratch.phi[j];
                        const auto& grad_j = scratch.grad_phi[j];

                        copy.cell_matrix(i, j) +=
                            (mu * grad_i * grad_j + (beta * grad_j) * phi_i +
                             gamma * phi_i * phi_j) *
                            JxW;
                    }
                    copy.cell_rhs(i) += f * phi_i * JxW;
                }
            }

            // Neumann Boundary Handling
            for (const auto& face : cell->face_iterators()) {
                if (face->at_boundary()) {
                    const auto bid = face->boundary_id();
                    if (const auto& neumann_ids = problem.get_neumann_ids();
                        neumann_ids.find(bid) != neumann_ids.end()) {
                        scratch.fe_face_values.reinit(cell, face);
                        const auto& fq_points =
                            scratch.fe_face_values.get_quadrature_points();
                        const unsigned int n_face_q = fq_points.size();

                        for (unsigned int q = 0; q < n_face_q; ++q) {
                            const double h =
                                problem.get_neumann_function(bid).value(
                                    fq_points[q]);
                            const double JxW = scratch.fe_face_values.JxW(q);

                            for (unsigned int k = 0; k < n_dofs; ++k) {
                                scratch.face_phi[k] =
                                    scratch.fe_face_values.shape_value(k, q);
                            }

                            for (unsigned int i = 0; i < n_dofs; ++i) {
                                copy.cell_rhs(i) +=
                                    h * scratch.face_phi[i] * JxW;
                            }
                        }
                    }
                }
            }

            cell->get_dof_indices(copy.local_dof_indices);
        };

    auto copier = [&](const CopyData& copy) {
        this->constraints.distribute_local_to_global(
            copy.cell_matrix, copy.cell_rhs, copy.local_dof_indices,
            system_matrix, system_rhs);
    };

    ScratchData<dim> scratch(*this->fe, QGauss<dim>(fe_degree + 1),
                             QGauss<dim - 1>(fe_degree + 1));

    WorkStream::run(filter_iterators(this->dof_handler.active_cell_iterators(),
                                     IteratorFilters::LocallyOwnedCell()),
                    worker, copier, scratch, CopyData());

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}

template <int dim> void MatrixBasedSolver<dim>::solve() {

    SolverControl solver_control(this->parameters.max_iterations,
                                 this->parameters.tolerance *
                                     system_rhs.l2_norm());
    LADistributed::MPI::Vector dist_solution(this->locally_owned_dofs,
                                             this->mpi_communicator);

    // CRITICAL FIX: Configure AMG for better parallel scaling
    LADistributed::MPI::PreconditionAMG prec;
    LADistributed::MPI::PreconditionAMG::AdditionalData amg_data;
    // Tuned settings for parallel scaling:
    amg_data.symmetric_operator = problem.is_symmetric();

    // These settings improve parallel scaling:
    // - Higher threshold reduces fill-in and communication
    // - More aggressive coarsening reduces levels
    amg_data.strong_threshold =
        0.5; // Default is 0.25, higher = less coupling = better parallel
    amg_data.aggressive_coarsening_num_levels =
        2; // Aggressive coarsening on first 2 levels

    prec.initialize(system_matrix, amg_data);

    if (problem.is_symmetric()) {
        LADistributed::SolverCG solver(solver_control);
        solver.solve(system_matrix, dist_solution, system_rhs, prec);
    } else {
        LADistributed::SolverGMRES solver(solver_control);
        solver.solve(system_matrix, dist_solution, system_rhs, prec);
    }

    this->constraints.distribute(dist_solution);
    solution = dist_solution;
    solution.update_ghost_values();

    this->timing_results.n_iterations = solver_control.last_step();

    if (this->parameters.verbose) {
        this->pcout << "   Converged in " << solver_control.last_step()
                    << " iterations." << std::endl;
    }
}

template <int dim> double MatrixBasedSolver<dim>::compute_l2_error() {
    if (!problem.has_exact_solution())
        return -1.0;

    Vector<double> cell_errors(this->triangulation.n_active_cells());

    VectorTools::integrate_difference(*this->mapping, this->dof_handler,
                                      solution, problem.get_exact_solution(),
                                      cell_errors, QGauss<dim>(fe_degree + 2),
                                      VectorTools::L2_norm);

    const double local_sq = cell_errors.norm_sqr();
    const double global_sq =
        Utilities::MPI::sum(local_sq, this->mpi_communicator);

    return std::sqrt(global_sq);
}

template <int dim>
void MatrixBasedSolver<dim>::output_results(unsigned int cycle) const {
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
    data_out.write_vtu_with_pvtu_record("./", this->parameters.output_prefix,
                                        cycle, this->mpi_communicator, 2);
}

template <int dim>
void MatrixBasedSolver<dim>::run(unsigned int n_refinements) {
    this->pcout << "Running: " << problem.get_name()
                << " (Matrix-Based, Hybrid MPI+Threading)" << std::endl;
    this->pcout << "   Using " << this->n_mpi_processes << " MPI processes"
                << std::endl;
    this->pcout << "   Threading: " << MultithreadInfo::n_threads()
                << " threads per process" << std::endl;

    MPI_Barrier(this->mpi_communicator);
    const auto t0 = std::chrono::high_resolution_clock::now();

    this->setup_grid(n_refinements);
    setup_dofs();

    MPI_Barrier(this->mpi_communicator);
    const auto t1 = std::chrono::high_resolution_clock::now();

    assemble_system();

    MPI_Barrier(this->mpi_communicator);
    const auto t2 = std::chrono::high_resolution_clock::now();

    solve();

    MPI_Barrier(this->mpi_communicator);
    const auto t3 = std::chrono::high_resolution_clock::now();

    output_results(0);
    double err = compute_l2_error();
    this->timing_results.memory_mb = compute_memory_usage();

    this->timing_results.setup_time =
        std::chrono::duration<double>(t1 - t0).count();
    this->timing_results.assembly_time =
        std::chrono::duration<double>(t2 - t1).count();
    this->timing_results.solve_time =
        std::chrono::duration<double>(t3 - t2).count();
    this->timing_results.total_time =
        std::chrono::duration<double>(t3 - t0).count();
    this->timing_results.n_dofs = this->dof_handler.n_dofs();
    this->timing_results.l2_error = err;
    this->timing_results.n_cells = this->triangulation.n_global_active_cells();

    if (this->parameters.verbose) {
        this->pcout << "   Setup time:    " << this->timing_results.setup_time
                    << "s" << std::endl;
        this->pcout << "   Assembly time: "
                    << this->timing_results.assembly_time << "s" << std::endl;
        this->pcout << "   Solve time:    " << this->timing_results.solve_time
                    << "s" << std::endl;
        this->pcout << "   Iterations:    " << this->timing_results.n_iterations
                    << std::endl;
        this->pcout << "   Memory usage:  " << this->timing_results.memory_mb
                    << " MB" << std::endl;
        this->pcout << "   L2 Error:      " << err << std::endl;
    }
}

// Explicit Instantiation
template class MatrixBasedSolver<2>;
template class MatrixBasedSolver<3>;

} // namespace HybridADRSolver