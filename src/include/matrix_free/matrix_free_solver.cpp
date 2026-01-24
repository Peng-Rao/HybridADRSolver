/**
 * @file matrix_free_solver.cpp
 * @brief Implementation of the Matrix-free solver.
 */

#include "matrix_free/matrix_free_solver.h"

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/solver_cg.h>
namespace HybridADRSolver {
using namespace dealii;

template <int dim, int fe_degree>
MatrixFreeSolver<dim, fe_degree>::MatrixFreeSolver(
    const ProblemInterface<dim>& problem, MPI_Comm comm,
    const SolverParameters& params)
    : ParallelSolverBase<dim>(comm, params), problem(problem) {
    // Initialize finite element and mapping
    this->fe = std::make_unique<FE_Q<dim>>(fe_degree);
    this->mapping = std::make_unique<MappingQ<dim>>(fe_degree);
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::setup_dofs() {
    // Distribute degrees of freedom
    this->dof_handler.distribute_dofs(*this->fe);

    // Extract locally owned and relevant DoFs
    this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    this->locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    // Setup constraints
    this->constraints.clear();
    this->constraints.reinit(this->locally_owned_dofs,
                             this->locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            this->constraints);

    // Apply Dirichlet boundary conditions
    for (const auto& id : problem.get_dirichlet_ids()) {
        VectorTools::interpolate_boundary_values(
            *this->mapping, this->dof_handler, id,
            problem.get_dirichlet_function(id), this->constraints);
    }
    this->constraints.close();

    // Setup MatrixFree
    setup_matrix_free();

    // Initialize vectors
    matrix_free_data->initialize_dof_vector(solution);
    matrix_free_data->initialize_dof_vector(system_rhs);

    if (this->parameters.verbose) {
        this->pcout << "   Number of DoFs: " << this->dof_handler.n_dofs()
                    << std::endl;
        this->pcout << "   Polynomial degree: " << fe_degree << std::endl;
        this->pcout << "   DoFs per cell: " << this->fe->n_dofs_per_cell()
                    << std::endl;
    }
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::setup_matrix_free() {
    // Configure MatrixFree for hybrid parallelization
    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    // Task parallelism settings for hybrid MPI+threading (TBB)
    additional_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::partition_partition;

    // Cell batching for SIMD vectorization
    additional_data.mapping_update_flags = update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points;

    // Boundary face integration flags (required for Neumann BCs)
    additional_data.mapping_update_flags_boundary_faces =
        update_values | update_JxW_values | update_quadrature_points |
        update_normal_vectors;

    // Overlap communication and computation
    additional_data.overlap_communication_computation = true;

    // Initialize MatrixFree
    matrix_free_data = std::make_shared<MatrixFree<dim, Number>>();
    matrix_free_data->reinit(*this->mapping, this->dof_handler,
                             this->constraints, QGauss<1>(fe_degree + 1),
                             additional_data);

    // Initialize the operator
    system_operator.initialize(matrix_free_data, problem);
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::assemble_system() {
    // Matrix-free: no system matrix to assemble
    // Only assemble the RHS
    assemble_rhs();
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::assemble_rhs() {
    system_rhs = 0;

    // Use FEEvaluation for efficient RHS assembly
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(
        *matrix_free_data);

    for (unsigned int cell = 0; cell < matrix_free_data->n_cell_batches();
         ++cell) {
        phi.reinit(cell);

        for (unsigned int q = 0; q < phi.n_q_points; ++q) {
            Point<dim, VectorizedArray<Number>> q_point =
                phi.quadrature_point(q);

            // Evaluate source term at quadrature points
            VectorizedArray<Number> f_val{};
            for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
                Point<dim> p;
                for (unsigned int d = 0; d < dim; ++d)
                    p[d] = q_point[d][v];
                f_val[v] = problem.source_term(p);
            }

            phi.submit_value(f_val, q);
        }

        phi.integrate_scatter(EvaluationFlags::values, system_rhs);
    }

    // Handle Neumann boundary conditions using face integration
    if (const auto& neumann_ids = problem.get_neumann_ids();
        !neumann_ids.empty()) {
        FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_face(
            *matrix_free_data, true);

        for (unsigned int face = matrix_free_data->n_inner_face_batches();
             face < matrix_free_data->n_inner_face_batches() +
                        matrix_free_data->n_boundary_face_batches();
             ++face) {

            if (const auto bid = matrix_free_data->get_boundary_id(face);
                neumann_ids.find(bid) != neumann_ids.end()) {
                phi_face.reinit(face);

                for (unsigned int q = 0; q < phi_face.n_q_points; ++q) {
                    auto q_point = phi_face.quadrature_point(q);

                    VectorizedArray<Number> neumann_val{};
                    for (unsigned int v = 0;
                         v < VectorizedArray<Number>::size(); ++v) {
                        Point<dim> p;
                        for (unsigned int d = 0; d < dim; ++d)
                            p[d] = q_point[d][v];
                        // Neumann term: μ * g_N where g_N = ∇u·n
                        const double mu = problem.diffusion_coefficient(p);
                        const double g_N =
                            problem.get_neumann_function(bid).value(p, 0);
                        neumann_val[v] = mu * g_N;
                    }

                    // Neumann contribution: mu * g_N
                    phi_face.submit_value(neumann_val, q);
                }

                phi_face.integrate_scatter(EvaluationFlags::values, system_rhs);
            }
        }
    }

    system_rhs.compress(VectorOperation::add);

    // Apply constraints to RHS
    this->constraints.set_zero(system_rhs);
}

// ============================================================================
// Solve Methods
// ============================================================================

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::solve() {
    if (problem.is_symmetric()) {
        solve_cg();
    } else {
        solve_gmres_jacobi();
    }

    // Update ghost values for output/error computation
    solution.update_ghost_values();
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::solve_cg() {
    SolverControl solver_control(this->parameters.max_iterations,
                                 this->parameters.tolerance *
                                     system_rhs.l2_norm());

    SolverCG<VectorType> solver(solver_control);

    // Setup Jacobi preconditioner
    JacobiPreconditioner<dim, fe_degree, Number> preconditioner;
    preconditioner.initialize(system_operator);

    // Solve
    solution = 0;
    solver.solve(system_operator, solution, system_rhs, preconditioner);

    // Apply constraints
    this->constraints.distribute(solution);

    if (this->parameters.verbose) {
        this->pcout << "   CG converged in " << solver_control.last_step()
                    << " iterations." << std::endl;
    }

    this->timing_results.n_iterations = solver_control.last_step();
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::solve_gmres_jacobi() {
    SolverControl solver_control(this->parameters.max_iterations,
                                 this->parameters.tolerance *
                                     system_rhs.l2_norm());

    SolverGMRES<VectorType>::AdditionalData gmres_data;
    gmres_data.max_n_tmp_vectors = 100;
    gmres_data.right_preconditioning = true;

    SolverGMRES<VectorType> solver(solver_control, gmres_data);

    // Setup Jacobi preconditioner
    JacobiPreconditioner<dim, fe_degree, Number> preconditioner;
    preconditioner.initialize(system_operator);

    // Solve
    solution = 0;
    solver.solve(system_operator, solution, system_rhs, preconditioner);

    // Apply constraints
    this->constraints.distribute(solution);

    if (this->parameters.verbose) {
        this->pcout << "   GMRES converged in " << solver_control.last_step()
                    << " iterations." << std::endl;
    }

    this->timing_results.n_iterations = solver_control.last_step();
}

template <int dim, int fe_degree>
double MatrixFreeSolver<dim, fe_degree>::compute_l2_error() const {
    if (!problem.has_exact_solution())
        return -1.0;

    double local_error_sq = 0.0;

    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(
        *matrix_free_data);

    for (unsigned int cell = 0; cell < matrix_free_data->n_cell_batches();
         ++cell) {
        phi.reinit(cell);
        phi.read_dof_values(solution);
        phi.evaluate(EvaluationFlags::values);

        VectorizedArray<Number> local_sum = {};

        for (unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto q_point = phi.quadrature_point(q);
            const VectorizedArray<Number> u_h = phi.get_value(q);

            VectorizedArray<Number> u_exact{};
            for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
                Point<dim> p;
                for (unsigned int d = 0; d < dim; ++d)
                    p[d] = q_point[d][v];
                u_exact[v] = problem.get_exact_solution().value(p, 0);
            }

            const VectorizedArray<Number> diff = u_h - u_exact;
            local_sum += diff * diff * phi.JxW(q);
        }

        // Sum over vector lanes
        for (unsigned int v = 0;
             v < matrix_free_data->n_active_entries_per_cell_batch(cell); ++v) {
            local_error_sq += local_sum[v];
        }
    }

    // Sum across MPI processes
    const double global_error_sq =
        Utilities::MPI::sum(local_error_sq, this->mpi_communicator);

    return std::sqrt(global_error_sq);
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::output_results(
    unsigned int cycle) const {
    if (!this->parameters.output_solution)
        return;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler);
    data_out.add_data_vector(solution, "solution");

    // Add subdomain info
    Vector<float> subdomain(this->triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = this->triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(*this->mapping);
    data_out.write_vtu_with_pvtu_record("./",
                                        this->parameters.output_prefix + "_mf",
                                        cycle, this->mpi_communicator, 2);
}

template <int dim, int fe_degree>
double MatrixFreeSolver<dim, fe_degree>::get_memory_usage() const {
    double memory = 0.0;

    // Solution and RHS vectors
    memory += solution.memory_consumption();
    memory += system_rhs.memory_consumption();

    // MatrixFree data structure
    memory += matrix_free_data->memory_consumption();

    // Convert to MB
    return memory / (1024.0 * 1024.0);
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::run(unsigned int n_ref) {
    this->pcout << "Running: " << problem.get_name()
                << " (Matrix-Free, Hybrid MPI+Threading)" << std::endl;
    this->pcout << "   Using " << this->n_mpi_processes << " MPI processes"
                << std::endl;
    this->pcout << "   Threading: " << MultithreadInfo::n_threads()
                << " threads per process" << std::endl;

    const auto t0 = std::chrono::high_resolution_clock::now();

    // Setup grid
    this->setup_grid(n_ref);

    // Distribute DoFs and setup MatrixFree
    setup_dofs();

    const auto t1 = std::chrono::high_resolution_clock::now();

    // Assemble RHS (no system matrix in matrix-free)
    assemble_system();

    const auto t2 = std::chrono::high_resolution_clock::now();

    // Solve linear system
    solve();

    const auto t3 = std::chrono::high_resolution_clock::now();

    // Output and error computation
    output_results(0);
    double err = compute_l2_error();

    // Store timing results
    this->timing_results.setup_time =
        std::chrono::duration<double>(t1 - t0).count();
    this->timing_results.assembly_time =
        std::chrono::duration<double>(t2 - t1).count();
    this->timing_results.solve_time =
        std::chrono::duration<double>(t3 - t2).count();
    this->timing_results.total_time =
        std::chrono::duration<double>(t3 - t0).count();
    this->timing_results.memory_mb = get_memory_usage();
    this->timing_results.n_dofs = this->dof_handler.n_dofs();

    if (this->parameters.verbose) {
        this->pcout << "   Setup time:    " << this->timing_results.setup_time
                    << "s" << std::endl;
        this->pcout << "   Assembly time: "
                    << this->timing_results.assembly_time << "s" << std::endl;
        this->pcout << "   Solve time:    " << this->timing_results.solve_time
                    << "s" << std::endl;
        this->pcout << "   Memory usage:  " << this->timing_results.memory_mb
                    << " MB" << std::endl;
        this->pcout << "   L2 Error:      " << err << std::endl;
    }
}

template class MatrixFreeSolver<2, 1>;
template class MatrixFreeSolver<2, 2>;
template class MatrixFreeSolver<2, 3>;
template class MatrixFreeSolver<3, 1>;
template class MatrixFreeSolver<3, 2>;
template class MatrixFreeSolver<3, 3>;
} // namespace HybridADRSolver