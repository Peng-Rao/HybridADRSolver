/**
 * @file solver.h
 * @brief Abstract base class for all solvers
 */
#ifndef HYBRIDADRSOLVER_SOLVER_H
#define HYBRIDADRSOLVER_SOLVER_H

#include "types.h"
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/affine_constraints.h>

namespace HybridADRSolver {
using namespace dealii;
template <int dim> class SolverBase {
public:
    /**
     * Constructor
     * @param comm MPI communicator
     * @param params Solver parameters
     */
    SolverBase(MPI_Comm comm, const SolverParameters& params);
    virtual ~SolverBase() = default;

    /**
     * Run the complete solve cycle
     * @param n_refinements Number of global mesh refinements
     */
    virtual void run(const unsigned int n_refinements) = 0;

    /**
     * Get timing results from the last run
     */
    const TimingResults& get_timing_results() const { return timing_results; }

    /**
     * Get the solver type (matrix-based or matrix-free)
     */
    virtual SolverType get_solver_type() const = 0;

    /**
     * Get a descriptive name for the solver
     */
    virtual std::string get_name() const = 0;

protected:
    /**
     * Setup the mesh/triangulation
     * @param n_refinements Number of global refinements
     */
    virtual void setup_grid(const unsigned int n_refinements);

    /**
     * Distribute degrees of freedom and setup constraints
     */
    virtual void setup_dofs() = 0;

    /**
     * Assemble the system (matrix and/or RHS)
     */
    virtual void assemble_system() = 0;

    /**
     * Solve the linear system
     */
    virtual void solve() = 0;

    /**
     * Output results to files
     * @param cycle Current refinement cycle
     */
    virtual void output_results(const unsigned int cycle) const = 0;

    // MPI communication
    MPI_Comm mpi_communicator;
    unsigned int n_mpi_processes;
    unsigned int this_mpi_process;

    // Mesh and finite element
    parallel::distributed::Triangulation<dim> triangulation;
    std::unique_ptr<FiniteElement<dim>> fe;
    DoFHandler<dim> dof_handler;
    std::unique_ptr<Mapping<dim>> mapping;

    // Constraints (Dirichlet BC, hanging nodes)
    AffineConstraints<double> constraints;

    // DoF distribution info
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    // Parameters
    SolverParameters parameters;

    // Timing
    TimingResults timing_results;

    // Output
    ConditionalOStream pcout;
    TimerOutput computing_timer;
};

template <int dim>
SolverBase<dim>::SolverBase(MPI_Comm comm, const SolverParameters& params)
    : mpi_communicator(comm),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(comm)),
      this_mpi_process(Utilities::MPI::this_mpi_process(comm)),
      triangulation(comm, typename Triangulation<dim>::MeshSmoothing(
                              Triangulation<dim>::smoothing_on_refinement |
                              Triangulation<dim>::smoothing_on_coarsening)),
      dof_handler(triangulation), parameters(params),
      pcout(std::cout, this_mpi_process == 0),
      computing_timer(comm, pcout, TimerOutput::never,
                      TimerOutput::wall_times) {
    // Set thread limit
    if (parameters.n_threads != numbers::invalid_unsigned_int)
        MultithreadInfo::set_thread_limit(parameters.n_threads);
}

template <int dim>
void SolverBase<dim>::setup_grid(const unsigned int n_refinements) {
    // Default: unit hypercube with boundary IDs
    GridGenerator::hyper_cube(triangulation, 0.0, 1.0, true);

    // Set boundary IDs: 0 for Dirichlet, 1 for Neumann
    for (auto& cell : triangulation.active_cell_iterators()) {
        for (const auto& face : cell->face_iterators()) {
            if (face->at_boundary()) {
                const Point<dim> center = face->center();
                // Dirichlet on x=0 face
                if (std::abs(center[0]) < 1e-12)
                    face->set_boundary_id(0);
                else
                    face->set_boundary_id(1);
            }
        }
    }

    triangulation.refine_global(n_refinements);

    if (parameters.verbose) {
        pcout << "   Active cells: " << triangulation.n_global_active_cells()
              << std::endl;
    }
}

} // namespace HybridADRSolver

#endif // HYBRIDADRSOLVER_SOLVER_H
