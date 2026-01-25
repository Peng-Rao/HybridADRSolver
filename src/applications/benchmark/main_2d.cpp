/**
 * @file main_2d.cpp
 * @brief 2D benchmark driver for comparing hybrid vs fully distributed solvers
 *
 * This benchmark compares:
 * 1. Matrix-based solver (MPI distributed with WorkStream threading)
 * 2. Matrix-free solver (Hybrid MPI + threading throughout)
 *
 * Tests: Time complexity analysis from refinement 2 to 10
 *
 * Usage:
 *   mpirun -np <N> ./benchmark_2d [options]
 */

#include "core/problem_definition.h"
#include "core/types.h"
#include "matrix_based/matrix_based_solver.h"
#include "matrix_free/matrix_free_solver.h"
#include "utils.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>

#include <getopt.h>
#include <iostream>
#include <string>

using namespace dealii;
using namespace HybridADRSolver;
using namespace BenchmarkUtils;

/**
 * @brief Run a single benchmark for the matrix-based solver (2D)
 */
template <int dim>
BenchmarkResult run_matrix_based_benchmark(const ProblemInterface<dim>& problem,
                                           int n_refinements, int degree,
                                           MPI_Comm comm,
                                           const std::string& test_type) {
    BenchmarkResult result;
    result.solver_type = "matrix_based";
    result.test_type = test_type;
    result.n_refinements = n_refinements;
    result.polynomial_degree = degree;

    MPI_Comm_size(comm, &result.n_mpi_processes);
    result.n_threads_per_process = MultithreadInfo::n_threads();

    SolverParameters params;
    params.verbose = false;
    params.output_solution = false;
    params.max_iterations = 2000;
    params.tolerance = 1e-10;

    BenchmarkUtils::Timer total_timer;
    total_timer.start();

    MatrixBasedSolver<dim> solver(problem, degree, comm, params);
    solver.run(n_refinements);

    total_timer.stop();

    const auto& timing = solver.get_timing_results();
    result.setup_time = timing.setup_time;
    result.assembly_time = timing.assembly_time;
    result.solve_time = timing.solve_time;
    result.total_time = total_timer.get_elapsed();
    result.n_iterations = timing.n_iterations;
    result.memory_mb = timing.memory_mb;
    result.n_dofs = timing.n_dofs;
    result.n_cells = timing.n_cells;
    result.l2_error = timing.l2_error;

    result.dofs_per_second =
        (result.total_time > 0) ? result.n_dofs / result.total_time : 0.0;

    return result;
}

/**
 * @brief Run a single benchmark for the matrix-free solver (2D)
 */
template <int dim, int fe_degree>
BenchmarkResult run_matrix_free_benchmark(const ProblemInterface<dim>& problem,
                                          int n_refinements, MPI_Comm comm,
                                          const std::string& test_type) {
    BenchmarkResult result;
    result.solver_type = "matrix_free";
    result.test_type = test_type;
    result.n_refinements = n_refinements;
    result.polynomial_degree = fe_degree;

    MPI_Comm_size(comm, &result.n_mpi_processes);
    result.n_threads_per_process = MultithreadInfo::n_threads();

    SolverParameters params;
    params.verbose = false;
    params.output_solution = false;
    params.solver_type = SolverType::MatrixFree;
    params.max_iterations = 2000;
    params.tolerance = 1e-10;

    BenchmarkUtils::Timer total_timer;
    total_timer.start();

    MatrixFreeSolver<dim, fe_degree> solver(problem, comm, params);
    solver.run(n_refinements);

    total_timer.stop();

    const auto& timing = solver.get_timing_results();
    result.setup_time = timing.setup_time;
    result.assembly_time = timing.assembly_time;
    result.solve_time = timing.solve_time;
    result.total_time = total_timer.get_elapsed();
    result.n_iterations = timing.n_iterations;
    result.memory_mb = timing.memory_mb;
    result.n_dofs = timing.n_dofs;
    result.n_cells = timing.n_cells;
    result.l2_error = timing.l2_error;

    result.dofs_per_second =
        (result.total_time > 0) ? result.n_dofs / result.total_time : 0.0;

    return result;
}

/**
 * @brief Run time complexity benchmark (refinement 2 to 10)
 */
void run_time_complexity_benchmark(ResultCollector& collector, int min_refs,
                                   int max_refs, int degree, MPI_Comm comm) {
    auto rank = Utilities::MPI::this_mpi_process(comm);
    auto n_procs = Utilities::MPI::n_mpi_processes(comm);
    auto n_threads = MultithreadInfo::n_threads();

    if (rank == 0) {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "TIME COMPLEXITY BENCHMARK (2D)\n";
        std::cout << std::string(80, '=') << "\n";
        std::cout << "MPI Processes: " << n_procs << "\n";
        std::cout << "Threads/Rank:  " << n_threads << "\n";
        std::cout << "Refinements:   " << min_refs << " to " << max_refs
                  << "\n";
        std::cout << "Degree:        " << degree << "\n";
        std::cout << std::string(80, '=') << "\n\n";
    }

    for (int refs = min_refs; refs <= max_refs; ++refs) {
        const Problems::ADRProblem<2> problem;
        if (rank == 0) {
            std::cout << "\n--- Refinement level " << refs << " ---\n";
        }

        // Matrix-based
        auto mb_result = run_matrix_based_benchmark<2>(problem, refs, degree,
                                                       comm, "time_complexity");
        collector.add_result(mb_result);

        // Matrix-free
        BenchmarkResult mf_result;
        if (degree == 1) {
            mf_result = run_matrix_free_benchmark<2, 1>(problem, refs, comm,
                                                        "time_complexity");
        } else if (degree == 2) {
            mf_result = run_matrix_free_benchmark<2, 2>(problem, refs, comm,
                                                        "time_complexity");
        } else if (degree == 3) {
            mf_result = run_matrix_free_benchmark<2, 3>(problem, refs, comm,
                                                        "time_complexity");
        }
        collector.add_result(mf_result);

        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  DoFs: " << mb_result.n_dofs
                      << ", Cells: " << mb_result.n_cells << "\n";
            std::cout << "  Matrix-Based: " << mb_result.total_time << "s, "
                      << mb_result.memory_mb << " MB\n";
            std::cout << "  Matrix-Free:  " << mf_result.total_time << "s, "
                      << mf_result.memory_mb << " MB\n";
            std::cout << "  Speedup: "
                      << mb_result.total_time / mf_result.total_time << "x\n";
        }
    }
}

void print_usage(const char* program_name) {
    std::cout
        << "Usage: mpirun -np <N> " << program_name << " [options]\n\n"
        << "Options:\n"
        << "  --min-ref, -m      Minimum refinements (default: 2)\n"
        << "  --max-ref, -M      Maximum refinements (default: 10)\n"
        << "  --degree, -d       Polynomial degree (default: 2)\n"
        << "  --output, -o       Output CSV file (default: "
           "complexity_2d.csv)\n"
        << "  --threads, -t      Number of threads per MPI process (default: "
           "auto)\n"
        << "  --help, -h         Show this help message\n";
}

int main(int argc, char* argv[]) {
    try {
        Utilities::MPI::MPI_InitFinalize mpi_init(
            argc, argv, numbers::invalid_unsigned_int);

        MPI_Comm comm = MPI_COMM_WORLD;
        const int rank = Utilities::MPI::this_mpi_process(comm);
        const int n_procs = Utilities::MPI::n_mpi_processes(comm);

        // Default options
        int min_refinements = 2;
        int max_refinements = 10;
        int degree = 2;
        int n_threads = -1; // Auto
        std::string output_file = "complexity_2d.csv";

        // Parse command line arguments
        static struct option long_options[] = {
            {"min-ref", required_argument, nullptr, 'm'},
            {"max-ref", required_argument, nullptr, 'M'},
            {"degree", required_argument, nullptr, 'd'},
            {"output", required_argument, nullptr, 'o'},
            {"threads", required_argument, nullptr, 't'},
            {"help", no_argument, nullptr, 'h'},
            {nullptr, 0, nullptr, 0}};

        int opt;
        while ((opt = getopt_long(argc, argv, "m:M:d:o:t:h", long_options,
                                  nullptr)) != -1) {
            switch (opt) {
                case 'm':
                    min_refinements = std::atoi(optarg);
                    break;
                case 'M':
                    max_refinements = std::atoi(optarg);
                    break;
                case 'd':
                    degree = std::atoi(optarg);
                    break;
                case 'o':
                    output_file = optarg;
                    break;
                case 't':
                    n_threads = std::atoi(optarg);
                    break;
                case 'h':
                    if (rank == 0)
                        print_usage(argv[0]);
                    return 0;
                default:
                    if (rank == 0)
                        print_usage(argv[0]);
                    return 1;
            }
        }

        if (n_threads > 0) {
            MultithreadInfo::set_thread_limit(n_threads);
        }

        if (rank == 0) {
            std::cout << "\n" << std::string(80, '*') << "\n";
            std::cout << "  2D TIME COMPLEXITY BENCHMARK\n";
            std::cout << "  Advection-Diffusion-Reaction Solver\n";
            std::cout << std::string(80, '*') << "\n\n";
            std::cout << "Configuration:\n";
            std::cout << "  MPI Processes:    " << n_procs << "\n";
            std::cout << "  Threads/Process:  " << MultithreadInfo::n_threads()
                      << "\n";
            std::cout << "  Total Cores:      "
                      << n_procs * MultithreadInfo::n_threads() << "\n";
            std::cout << "  Refinements:      " << min_refinements << " to "
                      << max_refinements << "\n";
            std::cout << "  Polynomial Degree: " << degree << "\n";
            std::cout << "  Output File:      " << output_file << "\n";
        }

        ResultCollector collector(output_file, comm);

        run_time_complexity_benchmark(collector, min_refinements,
                                      max_refinements, degree, comm);

        collector.write_csv();
        collector.print_summary();

        if (rank == 0) {
            std::cout << "\nResults written to: " << output_file << "\n";
        }

    } catch (std::exception& exc) {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception!" << std::endl;
        return 1;
    }

    return 0;
}