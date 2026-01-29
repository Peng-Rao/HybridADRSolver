/**
 * @file matrix_free_solver.cpp
 * @brief Implementation of the Matrix-free solver with GMG preconditioning.
 * * CRITICAL FIXES:
 * - MPI synchronization for accurate timing
 * - Works with fixed adr_operator.h that uses precomputed coefficients
 */

#include "matrix_free/matrix_free_solver.h"

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/solver_cg.h>

namespace HybridADRSolver {
using namespace dealii;

template <int dim, int fe_degree>
MatrixFreeSolver<dim, fe_degree>::MatrixFreeSolver(
    const ProblemInterface<dim>& problem, MPI_Comm comm,
    const SolverParameters& params)
    : ParallelSolverBase<dim>(comm, params, params.enable_multigrid),
      problem(problem) {
    this->fe = std::make_unique<FE_Q<dim>>(fe_degree);
    this->mapping = std::make_unique<MappingQ<dim>>(fe_degree);
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::setup_dofs() {
    this->dof_handler.distribute_dofs(*this->fe);
    this->dof_handler.distribute_mg_dofs();

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
            *this->mapping, this->dof_handler, id,
            problem.get_dirichlet_function(id), this->constraints);
    }
    this->constraints.close();

    setup_matrix_free();

    if (this->parameters.enable_multigrid) {
        setup_multigrid();
    }

    matrix_free_data->initialize_dof_vector(solution);
    matrix_free_data->initialize_dof_vector(system_rhs);

    if (this->parameters.verbose) {
        this->pcout << "   Number of DoFs: " << this->dof_handler.n_dofs()
                    << std::endl;
        this->pcout << "   Polynomial degree: " << fe_degree << std::endl;
        this->pcout << "   DoFs per cell: " << this->fe->n_dofs_per_cell()
                    << std::endl;
        this->pcout << "   Number of MG levels: "
                    << this->triangulation.n_global_levels() << std::endl;
    }
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::setup_matrix_free() {
    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    additional_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::partition_partition;

    additional_data.mapping_update_flags = update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points;

    additional_data.mapping_update_flags_boundary_faces =
        update_values | update_JxW_values | update_quadrature_points |
        update_normal_vectors;

    additional_data.overlap_communication_computation = true;

    matrix_free_data = std::make_shared<MatrixFree<dim, Number>>();
    matrix_free_data->reinit(*this->mapping, this->dof_handler,
                             this->constraints, QGauss<1>(fe_degree + 1),
                             additional_data);

    system_operator.initialize(matrix_free_data, problem);
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::setup_multigrid() {
    const unsigned int n_levels = this->triangulation.n_global_levels();

    mg_constrained_dofs.initialize(this->dof_handler);

    const std::set<types::boundary_id> dirichlet_ids =
        problem.get_dirichlet_ids();
    mg_constrained_dofs.make_zero_boundary_constraints(this->dof_handler,
                                                       dirichlet_ids);

    mg_matrices.resize(0, n_levels - 1);
    mg_matrix_free_storage.resize(0, n_levels - 1);

    for (unsigned int level = 0; level < n_levels; ++level) {
        AffineConstraints<double> level_constraints(
            this->dof_handler.locally_owned_mg_dofs(level),
            DoFTools::extract_locally_relevant_level_dofs(this->dof_handler,
                                                          level));

        for (const types::global_dof_index dof_index :
             mg_constrained_dofs.get_boundary_indices(level)) {
            level_constraints.constrain_dof_to_zero(dof_index);
        }
        level_constraints.close();

        typename MatrixFree<dim, LevelNumber>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
            MatrixFree<dim, LevelNumber>::AdditionalData::partition_partition;
        additional_data.mapping_update_flags =
            update_gradients | update_JxW_values | update_quadrature_points |
            update_values;
        additional_data.mg_level = level;
        additional_data.overlap_communication_computation = true;

        mg_matrix_free_storage[level] =
            std::make_shared<MatrixFree<dim, LevelNumber>>();
        mg_matrix_free_storage[level]->reinit(
            *this->mapping, this->dof_handler, level_constraints,
            QGauss<1>(fe_degree + 1), additional_data);

        mg_matrices[level].initialize(mg_matrix_free_storage[level],
                                      mg_constrained_dofs, level, problem);
    }

    if (this->parameters.verbose) {
        this->pcout << "   GMG hierarchy setup complete" << std::endl;
    }
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::assemble_system() {
    assemble_rhs();
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::assemble_rhs() {
    system_rhs = 0;

    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(
        *matrix_free_data);

    for (unsigned int cell = 0; cell < matrix_free_data->n_cell_batches();
         ++cell) {
        phi.reinit(cell);

        for (unsigned int q = 0; q < phi.n_q_points; ++q) {
            Point<dim, VectorizedArray<Number>> q_point =
                phi.quadrature_point(q);

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

    // Handle Neumann boundary conditions
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
                        const double mu = problem.diffusion_coefficient(p);
                        const double g_N =
                            problem.get_neumann_function(bid).value(p, 0);
                        neumann_val[v] = mu * g_N;
                    }

                    phi_face.submit_value(neumann_val, q);
                }

                phi_face.integrate_scatter(EvaluationFlags::values, system_rhs);
            }
        }
    }

    system_rhs.compress(VectorOperation::add);
    this->constraints.set_zero(system_rhs);
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::solve() {
    // Synchronize before timing the solve
    MPI_Barrier(this->mpi_communicator);

    if (this->parameters.enable_multigrid) {
        if (problem.is_symmetric()) {
            solve_cg_gmg();
        } else {
            solve_gmres_gmg();
        }
    } else {
        if (problem.is_symmetric()) {
            solve_cg_jacobi();
        } else {
            solve_gmres_jacobi();
        }
    }

    solution.update_ghost_values();
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::solve_cg_gmg() {
    MGTransferMatrixFree<dim, LevelNumber> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(this->dof_handler);

    using SmootherType =
        PreconditionChebyshev<LevelMatrixType, LevelVectorType>;
    mg::SmootherRelaxation<SmootherType, LevelVectorType> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, this->triangulation.n_global_levels() - 1);

    for (unsigned int level = 0; level < this->triangulation.n_global_levels();
         ++level) {
        if (level > 0) {
            smoother_data[level].smoothing_range = 15.0;
            smoother_data[level].degree = 5;
            smoother_data[level].eig_cg_n_iterations = 15;
        } else {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }

        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner =
            mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LevelVectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LevelVectorType> mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
        mg_interface_matrices;
    mg_interface_matrices.resize(0, this->triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < this->triangulation.n_global_levels();
         ++level) {
        mg_interface_matrices[level].initialize(mg_matrices[level]);
    }
    mg::Matrix<LevelVectorType> mg_interface(mg_interface_matrices);

    Multigrid<LevelVectorType> mg(mg_matrix, mg_coarse, mg_transfer,
                                  mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, LevelVectorType, MGTransferMatrixFree<dim, LevelNumber>>
        preconditioner(this->dof_handler, mg, mg_transfer);

    SolverControl solver_control(this->parameters.max_iterations,
                                 this->parameters.tolerance *
                                     system_rhs.l2_norm());
    SolverCG<VectorType> solver(solver_control);

    solution = 0;
    this->constraints.set_zero(solution);
    solver.solve(system_operator, solution, system_rhs, preconditioner);

    this->constraints.distribute(solution);

    if (this->parameters.verbose) {
        this->pcout << "   CG+GMG converged in " << solver_control.last_step()
                    << " iterations." << std::endl;
    }

    this->timing_results.n_iterations = solver_control.last_step();
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::solve_gmres_gmg() {
    MGTransferMatrixFree<dim, LevelNumber> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(this->dof_handler);

    using SmootherType =
        PreconditionChebyshev<LevelMatrixType, LevelVectorType>;
    mg::SmootherRelaxation<SmootherType, LevelVectorType> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, this->triangulation.n_global_levels() - 1);

    for (unsigned int level = 0; level < this->triangulation.n_global_levels();
         ++level) {
        if (level > 0) {
            smoother_data[level].smoothing_range = 20.0;
            smoother_data[level].degree = 5;
            smoother_data[level].eig_cg_n_iterations = 15;
        } else {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }

        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner =
            mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LevelVectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LevelVectorType> mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
        mg_interface_matrices;
    mg_interface_matrices.resize(0, this->triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < this->triangulation.n_global_levels();
         ++level) {
        mg_interface_matrices[level].initialize(mg_matrices[level]);
    }
    mg::Matrix<LevelVectorType> mg_interface(mg_interface_matrices);

    Multigrid<LevelVectorType> mg(mg_matrix, mg_coarse, mg_transfer,
                                  mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, LevelVectorType, MGTransferMatrixFree<dim, LevelNumber>>
        preconditioner(this->dof_handler, mg, mg_transfer);

    SolverControl solver_control(this->parameters.max_iterations,
                                 this->parameters.tolerance *
                                     system_rhs.l2_norm());

    SolverGMRES<VectorType>::AdditionalData gmres_data;
    gmres_data.max_n_tmp_vectors = 100;
    gmres_data.right_preconditioning = true;

    SolverGMRES<VectorType> solver(solver_control, gmres_data);

    solution = 0;
    this->constraints.set_zero(solution);
    solver.solve(system_operator, solution, system_rhs, preconditioner);

    this->constraints.distribute(solution);

    if (this->parameters.verbose) {
        this->pcout << "   GMRES+GMG converged in "
                    << solver_control.last_step() << " iterations."
                    << std::endl;
    }

    this->timing_results.n_iterations = solver_control.last_step();
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::solve_cg_jacobi() {
    SolverControl solver_control(this->parameters.max_iterations,
                                 this->parameters.tolerance *
                                     system_rhs.l2_norm());

    SolverCG<VectorType> solver(solver_control);

    system_operator.compute_diagonal();
    PreconditionJacobi<SystemMatrixType> preconditioner;
    preconditioner.initialize(system_operator);

    solution = 0;
    solver.solve(system_operator, solution, system_rhs, preconditioner);

    this->constraints.distribute(solution);

    if (this->parameters.verbose) {
        this->pcout << "   CG+Jacobi converged in "
                    << solver_control.last_step() << " iterations."
                    << std::endl;
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

    system_operator.compute_diagonal();
    PreconditionJacobi<SystemMatrixType> preconditioner;
    preconditioner.initialize(system_operator);

    solution = 0;
    solver.solve(system_operator, solution, system_rhs, preconditioner);

    this->constraints.distribute(solution);

    if (this->parameters.verbose) {
        this->pcout << "   GMRES+Jacobi converged in "
                    << solver_control.last_step() << " iterations."
                    << std::endl;
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

        for (unsigned int v = 0;
             v < matrix_free_data->n_active_entries_per_cell_batch(cell); ++v) {
            local_error_sq += local_sum[v];
        }
    }

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
double MatrixFreeSolver<dim, fe_degree>::compute_memory_usage() const {
    double memory = 0.0;

    memory += solution.memory_consumption();
    memory += system_rhs.memory_consumption();

    const double global_memory =
        Utilities::MPI::sum(memory, this->mpi_communicator);

    return global_memory / (1024.0 * 1024.0);
}

template <int dim, int fe_degree>
void MatrixFreeSolver<dim, fe_degree>::run(unsigned int n_ref) {
    this->pcout << "Running: " << problem.get_name()
                << " (Matrix-Free, GMG + Hybrid MPI+Threading)" << std::endl;
    this->pcout << "   Using " << this->n_mpi_processes << " MPI processes"
                << std::endl;
    this->pcout << "   Threading: " << MultithreadInfo::n_threads()
                << " threads per process" << std::endl;
    this->pcout << "   Multigrid: "
                << (this->parameters.enable_multigrid ? "Enabled" : "Disabled")
                << std::endl;

    // CRITICAL: Synchronize ALL processes before timing
    MPI_Barrier(this->mpi_communicator);
    const auto t0 = std::chrono::high_resolution_clock::now();

    this->setup_grid(n_ref);
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

    this->timing_results.setup_time =
        std::chrono::duration<double>(t1 - t0).count();
    this->timing_results.assembly_time =
        std::chrono::duration<double>(t2 - t1).count();
    this->timing_results.solve_time =
        std::chrono::duration<double>(t3 - t2).count();
    this->timing_results.total_time =
        std::chrono::duration<double>(t3 - t0).count();
    this->timing_results.memory_mb = compute_memory_usage();
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

// Explicit instantiations
template class MatrixFreeSolver<2, 1>;
template class MatrixFreeSolver<2, 2>;
template class MatrixFreeSolver<2, 3>;
template class MatrixFreeSolver<3, 1>;
template class MatrixFreeSolver<3, 2>;
template class MatrixFreeSolver<3, 3>;

} // namespace HybridADRSolver