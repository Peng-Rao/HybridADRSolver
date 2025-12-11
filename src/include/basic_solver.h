#ifndef HYBRIDADRSOLVER_GALERKINSOLVER_H
#define HYBRIDADRSOLVER_GALERKINSOLVER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#if defined(DEAL_II_WITH_PETSC) && defined(DEAL_II_WITH_MPI)
#define USE_PETSC_MPI
#include <deal.II/distributed/tria.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#else
#include <deal.II/grid/tria.h>
#endif

#include <iostream>

namespace BasicSolver {
using namespace dealii;

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

#ifdef USE_PETSC_MPI
    parallel::distributed::Triangulation<dim> triangulation;
    using MatrixType = PETScWrappers::MPI::SparseMatrix;
    using VectorType = PETScWrappers::MPI::Vector;
#else
    Triangulation<dim> triangulation;
    using MatrixType = SparseMatrix<double>;
    using VectorType = Vector<double>;
#endif

    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    MatrixType system_matrix;
    VectorType locally_relevant_solution;
    VectorType system_rhs;

    ConditionalOStream pcout{std::cout, true};

    const types::boundary_id dirichlet_boundary_id = 0;
    const types::boundary_id neumann_boundary_id = 1;
};
} // namespace BasicSolver

#endif
