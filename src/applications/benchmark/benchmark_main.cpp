/**
 * @file benchmark_main.cpp
 * @brief Main benchmark driver for comparing hybrid vs fully distributed
 * solvers
 *
 * This benchmark compares:
 * 1. Matrix-based solver (MPI distributed with WorkStream threading for
 * assembly)
 * 2. Matrix-free solver (Hybrid MPI + threading throughout)
 *
 * Tests performed:
 * - Strong scaling: Fixed problem size, varying number of processes/threads
 * - Weak scaling: Problem size scales with number of processes
 * - Memory efficiency comparison
 * - Throughput analysis (DoFs/second)
 *
 * Usage:
 *   mpirun -np <N> ./benchmark [options]
 *
 * Options:
 *   --strong       Run strong scaling test
 *   --weak         Run weak scaling test
 *   --refinements  Number of mesh refinements (default: 4)
 *   --degree       Polynomial degree (default: 2)
 *   --output        CSV file (default: benchmark_results.csv)
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
 * @brief Run a single benchmark for the matrix-based solver
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

    // Configure solver (minimal output during benchmark)
    SolverParameters params;
    params.verbose = false;
    params.output_solution = false;
    params.max_iterations = 2000;
    params.tolerance = 1e-10;

    BenchmarkUtils::Timer total_timer;
    total_timer.start();

    // Create and run solver
    MatrixBasedSolver<dim> solver(problem, degree, comm, params);
    solver.run(n_refinements);

    total_timer.stop();

    // Collect results
    const auto& timing = solver.get_timing_results();
    result.setup_time = timing.setup_time;
    result.assembly_time = timing.assembly_time;
    result.solve_time = timing.solve_time;
    result.total_time = total_timer.get_elapsed();
    result.n_iterations = timing.n_iterations;
    result.memory_mb = timing.memory_mb;
    result.n_dofs = timing.n_dofs;

    // Compute throughput
    result.dofs_per_second =
        (result.total_time > 0) ? result.n_dofs / result.total_time : 0.0;

    return result;
}

/**
 * @brief Run a single benchmark for the matrix-free solver
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

    // Configure solver
    SolverParameters params;
    params.verbose = false;
    params.output_solution = false;
    params.solver_type = SolverType::MatrixFree;
    params.max_iterations = 2000;
    params.tolerance = 1e-10;

    BenchmarkUtils::Timer total_timer;
    total_timer.start();

    // Create and run solver
    MatrixFreeSolver<dim, fe_degree> solver(problem, comm, params);
    solver.run(n_refinements);

    total_timer.stop();

    // Collect results
    const auto& timing = solver.get_timing_results();
    result.setup_time = timing.setup_time;
    result.assembly_time = timing.assembly_time;
    result.solve_time = timing.solve_time;
    result.total_time = total_timer.get_elapsed();
    result.n_iterations = timing.n_iterations;
    result.memory_mb = timing.memory_mb;
    result.n_dofs = timing.n_dofs;

    // Compute throughput
    result.dofs_per_second =
        (result.total_time > 0) ? result.n_dofs / result.total_time : 0.0;

    return result;
}

/**
 * @brief Run strong scaling benchmark
 * Fixed problem size, varying number of processes
 */
void run_strong_scaling_benchmark(ResultCollector& collector, int n_refinements,
                                  int degree, MPI_Comm comm) {
    const int rank = Utilities::MPI::this_mpi_process(comm);

    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "STRONG SCALING BENCHMARK\n";
        std::cout << "Refinements: " << n_refinements << ", Degree: " << degree
                  << "\n";
        std::cout << std::string(60, '=') << "\n";
    }

    const Problems::ADRProblem<3> problem;

    // Run matrix-based benchmark
    if (rank == 0)
        std::cout << "\nRunning Matrix-Based Solver...\n";
    auto mb_result = run_matrix_based_benchmark<3>(
        problem, n_refinements, degree, comm, "strong_scaling");
    collector.add_result(mb_result);

    // Run matrix-free benchmark (degree 2)
    if (rank == 0)
        std::cout << "Running Matrix-Free Solver...\n";
    BenchmarkResult mf_result;
    if (degree == 1) {
        mf_result = run_matrix_free_benchmark<3, 1>(problem, n_refinements,
                                                    comm, "strong_scaling");
    } else if (degree == 2) {
        mf_result = run_matrix_free_benchmark<3, 2>(problem, n_refinements,
                                                    comm, "strong_scaling");
    } else if (degree == 3) {
        mf_result = run_matrix_free_benchmark<3, 3>(problem, n_refinements,
                                                    comm, "strong_scaling");
    }
    collector.add_result(mf_result);

    // Print comparison
    if (rank == 0) {
        std::cout << "\nComparison:\n";
        std::cout << "  Matrix-Based: " << mb_result.total_time << "s, "
                  << mb_result.memory_mb << " MB\n";
        std::cout << "  Matrix-Free:  " << mf_result.total_time << "s, "
                  << mf_result.memory_mb << " MB\n";
        std::cout << "  Speedup (MF/MB): "
                  << mb_result.total_time / mf_result.total_time << "x\n";
        std::cout << "  Memory Ratio: "
                  << mb_result.memory_mb / mf_result.memory_mb << "x\n";
    }
}

/**
 * @brief Run weak scaling benchmark
 * Problem size scales with number of processes (constant work per process)
 */
void run_weak_scaling_benchmark(ResultCollector& collector,
                                int base_refinements, int degree,
                                MPI_Comm comm) {
    const int rank = Utilities::MPI::this_mpi_process(comm);
    const int n_procs = Utilities::MPI::n_mpi_processes(comm);

    // For weak scaling, increase refinements based on number of processes
    // Each doubling of processes should handle 2^dim more cells
    // So we add log2(n_procs)/dim refinements
    int extra_refs = 0;
    int temp = n_procs;
    while (temp > 1) {
        extra_refs++;
        temp /= 2;
    }
    const int n_refinements = base_refinements + extra_refs / 3;

    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "WEAK SCALING BENCHMARK\n";
        std::cout << "Base Refinements: " << base_refinements
                  << ", Actual: " << n_refinements << "\n";
        std::cout << "Processes: " << n_procs << ", Degree: " << degree << "\n";
        std::cout << std::string(60, '=') << "\n";
    }

    const Problems::ADRProblem<3> problem;

    // Run matrix-based benchmark
    if (rank == 0)
        std::cout << "\nRunning Matrix-Based Solver...\n";
    auto mb_result = run_matrix_based_benchmark<3>(
        problem, n_refinements, degree, comm, "weak_scaling");
    collector.add_result(mb_result);

    // Run matrix-free benchmark
    if (rank == 0)
        std::cout << "Running Matrix-Free Solver...\n";
    BenchmarkResult mf_result;
    if (degree == 2) {
        mf_result = run_matrix_free_benchmark<3, 2>(problem, n_refinements,
                                                    comm, "weak_scaling");
    }
    collector.add_result(mf_result);
}

/**
 * @brief Run memory efficiency benchmark at various problem sizes
 */
void run_memory_benchmark(ResultCollector& collector, int min_refs,
                          int max_refs, int degree, MPI_Comm comm) {
    const int rank = Utilities::MPI::this_mpi_process(comm);

    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "MEMORY EFFICIENCY BENCHMARK\n";
        std::cout << std::string(60, '=') << "\n";
    }

    for (int refs = min_refs; refs <= max_refs; ++refs) {
        const Problems::ADRProblem<3> problem;
        if (rank == 0) {
            std::cout << "\n--- Refinement level " << refs << " ---\n";
        }

        // Matrix-based
        auto mb_result = run_matrix_based_benchmark<3>(problem, refs, degree,
                                                       comm, "memory_test");
        collector.add_result(mb_result);

        // Matrix-free
        BenchmarkResult mf_result;
        if (degree == 2) {
            mf_result = run_matrix_free_benchmark<3, 2>(problem, refs, comm,
                                                        "memory_test");
        }
        collector.add_result(mf_result);

        if (rank == 0) {
            std::cout << "  DoFs: " << mb_result.n_dofs << "\n";
            std::cout << "  Matrix-Based Memory: " << mb_result.memory_mb
                      << " MB\n";
            std::cout << "  Matrix-Free Memory:  " << mf_result.memory_mb
                      << " MB\n";
            std::cout << "  Ratio (MB/MF): "
                      << mb_result.memory_mb / mf_result.memory_mb << "x\n";
        }
    }
}

/**
 * @brief Print usage information
 */
void print_usage(const char* program_name) {
    std::cout
        << "Usage: mpirun -np <N> " << program_name << " [options]\n\n"
        << "Options:\n"
        << "  --strong, -s       Run strong scaling test\n"
        << "  --weak, -w         Run weak scaling test\n"
        << "  --memory, -m       Run memory efficiency test\n"
        << "  --all, -a          Run all tests (default)\n"
        << "  --refinements, -r  Number of mesh refinements (default: 4)\n"
        << "  --degree, -d       Polynomial degree (default: 2)\n"
        << "  --output, -o       Output CSV file (default: "
           "benchmark_results.csv)\n"
        << "  --threads, -t      Number of threads per MPI process (default: "
           "auto)\n"
        << "  --help, -h         Show this help message\n";
}

int main(int argc, char* argv[]) {
    try {
        // Initialize MPI with threading support
        Utilities::MPI::MPI_InitFinalize mpi_init(
            argc, argv, numbers::invalid_unsigned_int);

        MPI_Comm comm = MPI_COMM_WORLD;
        const int rank = Utilities::MPI::this_mpi_process(comm);
        const int n_procs = Utilities::MPI::n_mpi_processes(comm);

        // Default options
        bool run_strong = false;
        bool run_weak = false;
        bool run_memory = false;
        bool run_all = true;
        int n_refinements = 4;
        int degree = 2;
        int n_threads = -1; // Auto
        std::string output_file = "benchmark_results.csv";

        // Parse command line arguments
        static struct option long_options[] = {
            {"strong", no_argument, nullptr, 's'},
            {"weak", no_argument, nullptr, 'w'},
            {"memory", no_argument, nullptr, 'm'},
            {"all", no_argument, nullptr, 'a'},
            {"refinements", required_argument, nullptr, 'r'},
            {"degree", required_argument, nullptr, 'd'},
            {"output", required_argument, nullptr, 'o'},
            {"threads", required_argument, nullptr, 't'},
            {"help", no_argument, nullptr, 'h'},
            {nullptr, 0, nullptr, 0}};

        int opt;
        while ((opt = getopt_long(argc, argv, "swmar:d:o:t:h", long_options,
                                  nullptr)) != -1) {
            switch (opt) {
                case 's':
                    run_strong = true;
                    run_all = false;
                    break;
                case 'w':
                    run_weak = true;
                    run_all = false;
                    break;
                case 'm':
                    run_memory = true;
                    run_all = false;
                    break;
                case 'a':
                    run_all = true;
                    break;
                case 'r':
                    n_refinements = std::atoi(optarg);
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

        // Set thread limit if specified
        if (n_threads > 0) {
            MultithreadInfo::set_thread_limit(n_threads);
        }

        // Print header
        if (rank == 0) {
            std::cout << "\n" << std::string(70, '*') << "\n";
            std::cout << "  HYBRID vs DISTRIBUTED PARALLELIZATION BENCHMARK\n";
            std::cout << "  Advection-Diffusion-Reaction Solver Comparison\n";
            std::cout << std::string(70, '*') << "\n\n";
            std::cout << "Configuration:\n";
            std::cout << "  MPI Processes:    " << n_procs << "\n";
            std::cout << "  Threads/Process:  " << MultithreadInfo::n_threads()
                      << "\n";
            std::cout << "  Total Cores:      "
                      << n_procs * MultithreadInfo::n_threads() << "\n";
            std::cout << "  Mesh Refinements: " << n_refinements << "\n";
            std::cout << "  Polynomial Degree: " << degree << "\n";
            std::cout << "  Output File:      " << output_file << "\n";
        }

        // Create result collector
        ResultCollector collector(output_file, comm);

        // Run requested benchmarks
        if (run_all || run_strong) {
            run_strong_scaling_benchmark(collector, n_refinements, degree,
                                         comm);
        }

        if (run_all || run_weak) {
            run_weak_scaling_benchmark(collector, n_refinements - 1, degree,
                                       comm);
        }

        if (run_all || run_memory) {
            run_memory_benchmark(collector, 2, std::min(n_refinements, 5),
                                 degree, comm);
        }

        // Output results
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
