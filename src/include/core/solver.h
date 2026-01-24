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

#include <utility>

namespace HybridADRSolver {
using namespace dealii;
template <int dim> class ParallelSolverBase {
public:
    /**
     * Constructor
     * @param comm MPI communicator
     * @param params Solver parameters
     */
    ParallelSolverBase(MPI_Comm comm, SolverParameters params);
    virtual ~ParallelSolverBase() = default;

    /**
     * Run the complete solve cycle
     * @param n_refinements Number of global mesh refinements
     */
    virtual void run(unsigned int n_refinements) = 0;

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
    virtual void setup_grid(unsigned int n_refinements);

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
    virtual void output_results(unsigned int cycle) const = 0;

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
ParallelSolverBase<dim>::ParallelSolverBase(MPI_Comm comm,
                                            SolverParameters params)
    : mpi_communicator(comm),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(comm)),
      this_mpi_process(Utilities::MPI::this_mpi_process(comm)),
      triangulation(comm, typename Triangulation<dim>::MeshSmoothing(
                              Triangulation<dim>::smoothing_on_refinement |
                              Triangulation<dim>::smoothing_on_coarsening)),
      dof_handler(triangulation), parameters(std::move(params)),
      pcout(std::cout, this_mpi_process == 0),
      computing_timer(comm, pcout, TimerOutput::never,
                      TimerOutput::wall_times) {
    // Set thread limit
    if (parameters.n_threads != numbers::invalid_unsigned_int)
        MultithreadInfo::set_thread_limit(parameters.n_threads);
}

template <int dim>
void ParallelSolverBase<dim>::setup_grid(const unsigned int n_refinements) {
    GridGenerator::hyper_cube(triangulation, 0.0, 1.0, true);

    triangulation.refine_global(n_refinements);

    if (parameters.verbose) {
        pcout << "   Active cells: " << triangulation.n_global_active_cells()
              << std::endl;
    }
}

} // namespace HybridADRSolver

#endif // HYBRIDADRSOLVER_SOLVER_H
