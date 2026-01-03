#include "matrix_based/matrix_based_solver.h"
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

namespace HybridADRSolver {

using namespace dealii;

template <int dim> struct ScratchData {
    ScratchData(const FiniteElement<dim>& fe, const Quadrature<dim>& q, const Quadrature<dim-1>& fq)
        : fe_values(fe, q, update_values | update_gradients | update_quadrature_points | update_JxW_values),
          fe_face_values(fe, fq, update_values | update_quadrature_points | update_JxW_values) {}

    ScratchData(const ScratchData& sd)
        : fe_values(sd.fe_values.get_fe(), sd.fe_values.get_quadrature(), sd.fe_values.get_update_flags()),
          fe_face_values(sd.fe_face_values.get_fe(), sd.fe_face_values.get_quadrature(), sd.fe_face_values.get_update_flags()) {}

    FEValues<dim> fe_values;
    FEFaceValues<dim> fe_face_values;
};

struct CopyData {
    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
};

template <int dim>
MatrixBasedSolver<dim>::MatrixBasedSolver(const ProblemInterface<dim>& prob, unsigned int degree, MPI_Comm comm, const SolverParameters& params)
    : ParallelSolverBase<dim>(comm, params), problem(prob), fe_degree(degree) {
    this->fe = std::make_unique<FE_Q<dim>>(degree);
    this->mapping = std::make_unique<MappingQ<dim>>(degree); // Initialize mapping
}

template <int dim>
void MatrixBasedSolver<dim>::setup_dofs() {
    this->dof_handler.distribute_dofs(*this->fe);

    DoFRenumbering::Cuthill_McKee(this->dof_handler);

    this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    this->locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    this->constraints.clear();
    this->constraints.reinit(this->locally_owned_dofs, this->locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);

    for (const auto& id : problem.get_dirichlet_ids()) {
        VectorTools::interpolate_boundary_values(this->dof_handler, id,
                                                 problem.get_dirichlet_function(id),
                                                 this->constraints);
    }
    this->constraints.close();

    DynamicSparsityPattern dsp(this->locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(this->dof_handler, dsp, this->constraints, false);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                               this->locally_owned_dofs,
                                               this->mpi_communicator,
                                               this->locally_relevant_dofs);

    system_matrix.reinit(this->locally_owned_dofs, this->locally_owned_dofs, dsp, this->mpi_communicator);
    system_rhs.reinit(this->locally_owned_dofs, this->mpi_communicator);
    solution.reinit(this->locally_owned_dofs, this->locally_relevant_dofs, this->mpi_communicator);

    if (this->parameters.verbose) {
        this->pcout << "   Number of DoFs: " << this->dof_handler.n_dofs() << std::endl;
    }
}

template <int dim>
void MatrixBasedSolver<dim>::assemble_system() {
    auto worker = [&](const typename DoFHandler<dim>::active_cell_iterator& cell, ScratchData<dim>& scratch, CopyData& copy) {
        copy.local_dof_indices.clear();

        // Safety check for distributed triangulation
        if (!cell->is_locally_owned()) return;

        const unsigned int n_dofs = scratch.fe_values.get_fe().n_dofs_per_cell();

        copy.cell_matrix.reinit(n_dofs, n_dofs);
        copy.cell_rhs.reinit(n_dofs);

        // --- FIX: Resize the vector before calling get_dof_indices ---
        copy.local_dof_indices.resize(n_dofs);
        // -------------------------------------------------------------

        scratch.fe_values.reinit(cell);

        const auto& q_points = scratch.fe_values.get_quadrature_points();
        const unsigned int n_q_points = q_points.size();

        for(unsigned int q=0; q<n_q_points; ++q) {
            const double mu = problem.diffusion_coefficient(q_points[q]);
            const auto beta = problem.advection_field(q_points[q]);
            const double gamma = problem.reaction_coefficient(q_points[q]);
            const double f = problem.source_term(q_points[q]);
            const double JxW = scratch.fe_values.JxW(q);

            for(unsigned int i=0; i<n_dofs; ++i) {
                const double phi_i = scratch.fe_values.shape_value(i, q);
                const auto grad_i = scratch.fe_values.shape_grad(i, q);

                for(unsigned int j=0; j<n_dofs; ++j) {
                    const double phi_j = scratch.fe_values.shape_value(j, q);
                    const auto grad_j = scratch.fe_values.shape_grad(j, q);

                    // Assembly: (mu*grad_u, grad_v) + (beta.grad_u, v) + (gamma*u, v)
                    copy.cell_matrix(i,j) += (mu * grad_i * grad_j + (beta * grad_j) * phi_i + gamma * phi_i * phi_j) * JxW;
                }
                copy.cell_rhs(i) += f * phi_i * JxW;
            }
        }

        // [Decoupled] Neumann Boundary Handling
        for(const auto& face : cell->face_iterators()) {
            if(face->at_boundary()) {
                const auto bid = face->boundary_id();
                const auto& neumann_ids = problem.get_neumann_ids();

                if(neumann_ids.find(bid) != neumann_ids.end()) {
                    scratch.fe_face_values.reinit(cell, face);
                    const auto& fq_points = scratch.fe_face_values.get_quadrature_points();
                    const unsigned int n_face_q = fq_points.size();

                    for(unsigned int q=0; q<n_face_q; ++q) {
                        const double h = problem.get_neumann_function(bid).value(fq_points[q]);
                        const double JxW = scratch.fe_face_values.JxW(q);

                        for(unsigned int i=0; i<n_dofs; ++i)
                            copy.cell_rhs(i) += h * scratch.fe_face_values.shape_value(i, q) * JxW;
                    }
                }
            }
        }

        cell->get_dof_indices(copy.local_dof_indices);
    };

    // WorkStream copier lambda
    auto copier = [&](const CopyData& copy) {
        // If vector is empty (e.g. not locally owned), skip
        if (copy.local_dof_indices.empty()) return;

        this->constraints.distribute_local_to_global(copy.cell_matrix,
                                                     copy.cell_rhs,
                                                     copy.local_dof_indices,
                                                     system_matrix,
                                                     system_rhs);
    };

    ScratchData<dim> scratch(*this->fe, QGauss<dim>(fe_degree+1), QGauss<dim-1>(fe_degree+1));

    WorkStream::run(this->dof_handler.begin_active(),
                    this->dof_handler.end(),
                    worker,
                    copier,
                    scratch,
                    CopyData());

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}

template <int dim>
void MatrixBasedSolver<dim>::solve() {
    SolverControl solver_control(this->parameters.max_iterations, this->parameters.tolerance * system_rhs.l2_norm());
    LADistributed::MPI::Vector dist_solution(this->locally_owned_dofs, this->mpi_communicator);

    // [Decoupled] Select solver based on symmetry
    if(problem.is_symmetric()) {
        LADistributed::SolverCG solver(solver_control);
        LADistributed::MPI::PreconditionAMG prec;
        prec.initialize(system_matrix);
        solver.solve(system_matrix, dist_solution, system_rhs, prec);
    } else {
        LADistributed::SolverGMRES solver(solver_control);
        LADistributed::MPI::PreconditionJacobi prec;
        prec.initialize(system_matrix);
        solver.solve(system_matrix, dist_solution, system_rhs, prec);
    }

    this->constraints.distribute(dist_solution);
    solution = dist_solution;
    solution.update_ghost_values();

    if(this->parameters.verbose) {
        this->pcout << "   Converged in " << solver_control.last_step() << " iterations." << std::endl;
    }
}

template <int dim>
double MatrixBasedSolver<dim>::compute_l2_error() {
    if(!problem.has_exact_solution()) return -1.0;

    Vector<double> cell_errors(this->triangulation.n_active_cells());

    VectorTools::integrate_difference(*this->mapping,
                                      this->dof_handler,
                                      solution,
                                      problem.get_exact_solution(),
                                      cell_errors,
                                      QGauss<dim>(fe_degree+2),
                                      VectorTools::L2_norm);

    const double local_sq = cell_errors.norm_sqr();
    const double global_sq = Utilities::MPI::sum(local_sq, this->mpi_communicator);

    return std::sqrt(global_sq);
}

template <int dim>
void MatrixBasedSolver<dim>::output_results(unsigned int cycle) const {
    if (!this->parameters.output_solution) return;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler);
    data_out.add_data_vector(solution, "solution");

    // Add subdomain info for verification
    Vector<float> subdomain(this->triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = this->triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(*this->mapping);
    data_out.write_vtu_with_pvtu_record("./", this->parameters.output_prefix, cycle, this->mpi_communicator, 2);
}

template <int dim>
void MatrixBasedSolver<dim>::run(unsigned int n_ref) {
    this->pcout << "Running: " << problem.get_name() << std::endl;

    const auto t0 = std::chrono::high_resolution_clock::now();

    this->setup_grid(n_ref);
    setup_dofs();

    const auto t1 = std::chrono::high_resolution_clock::now();
    assemble_system();

    const auto t2 = std::chrono::high_resolution_clock::now();
    solve();

    const auto t3 = std::chrono::high_resolution_clock::now();

    output_results(0);
    double err = compute_l2_error();

    if (this->parameters.verbose) {
        this->pcout << "   Setup time:    " << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;
        this->pcout << "   Assembly time: " << std::chrono::duration<double>(t2 - t1).count() << "s" << std::endl;
        this->pcout << "   Solve time:    " << std::chrono::duration<double>(t3 - t2).count() << "s" << std::endl;
        this->pcout << "   L2 Error:      " << err << std::endl;
    }
}

// Explicit Instantiation
template class MatrixBasedSolver<2>;
template class MatrixBasedSolver<3>;

} // namespace HybridADRSolver