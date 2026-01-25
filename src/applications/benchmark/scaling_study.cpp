/**
 * @file scaling_study.cpp
 * @brief Comprehensive strong and weak scaling benchmark for the Hybrid ADR
 * Solver
 *
 * This benchmark measures:
 * 1. Strong Scaling: Fixed problem size, varying core count
 *    - Speedup: S(p) = T(1) / T(p)
 *    - Efficiency: E(p) = S(p) / p = T(1) / (p * T(p))
 *
 * 2. Weak Scaling: Problem size scales with core count (constant work per core)
 *    - Efficiency: E(p) = T(1) / T(p)  (ideal = 1.0)
 *
 * Usage:
 *   mpirun -np <N> ./scaling_benchmark [options]
 *
 * Options:
 *   --strong          Run strong scaling test only
 *   --weak            Run weak scaling test only
 *   --min-ref <n>     Minimum refinements (default: 3)
 *   --max-ref <n>     Maximum refinements (default: 7)
 *   --degree <n>      Polynomial degree (default: 2)
 *   --output <file>   Output CSV file prefix (default: scaling_results)
 *   --threads <n>     Threads per MPI process (default: auto)
 *   --trials <n>      Number of trials for averaging (default: 3)
 *   --warmup <n>      Number of warmup runs (default: 1)
 *   --dim <n>         Dimension 2 or 3 (default: 2)
 */

#include "core/problem_definition.h"
#include "core/types.h"
#include "matrix_based/matrix_based_solver.h"
#include "matrix_free/matrix_free_solver.h"
#include "utils.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>

#include <algorithm>
#include <cmath>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace dealii;
using namespace HybridADRSolver;
using namespace BenchmarkUtils;

/**
 * @brief Extended benchmark result with scaling metrics
 */
struct ScalingResult {
    // Basic info
    std::string solver_type;
    std::string test_type;
    int dimension;
    int n_mpi_processes;
    int n_threads_per_process;
    int total_cores;
    int n_refinements;
    int polynomial_degree;

    // Problem size
    unsigned int n_dofs;
    unsigned int n_cells;

    // Timing (averaged over trials)
    double setup_time_avg;
    double setup_time_std;
    double assembly_time_avg;
    double assembly_time_std;
    double solve_time_avg;
    double solve_time_std;
    double total_time_avg;
    double total_time_std;

    // Solver stats
    double n_iterations_avg;
    double l2_error_avg;
    double memory_mb_avg;

    // Scaling metrics
    double speedup;             // T(1) / T(p) for strong scaling
    double parallel_efficiency; // S(p) / p for strong, T(1)/T(p) for weak
    double dofs_per_second;

    /**
     * @brief CSV header
     */
    static std::string csv_header() {
        return "solver_type,test_type,dimension,n_mpi,n_threads,total_cores,"
               "n_refinements,poly_degree,n_dofs,n_cells,"
               "setup_time_avg,setup_time_std,"
               "assembly_time_avg,assembly_time_std,"
               "solve_time_avg,solve_time_std,"
               "total_time_avg,total_time_std,"
               "n_iterations_avg,l2_error,memory_mb,"
               "speedup,parallel_efficiency,dofs_per_second";
    }

    /**
     * @brief Convert to CSV line
     */
    std::string to_csv() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << solver_type << "," << test_type << "," << dimension << ","
            << n_mpi_processes << "," << n_threads_per_process << ","
            << total_cores << "," << n_refinements << "," << polynomial_degree
            << "," << n_dofs << "," << n_cells << "," << setup_time_avg << ","
            << setup_time_std << "," << assembly_time_avg << ","
            << assembly_time_std << "," << solve_time_avg << ","
            << solve_time_std << "," << total_time_avg << "," << total_time_std
            << "," << n_iterations_avg << "," << l2_error_avg << ","
            << memory_mb_avg << "," << speedup << "," << parallel_efficiency
            << "," << dofs_per_second;
        return oss.str();
    }
};

/**
 * @brief Statistics helper
 */
struct Statistics {
    static double mean(const std::vector<double>& v) {
        if (v.empty())
            return 0.0;
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    }

    static double stddev(const std::vector<double>& v) {
        if (v.size() < 2)
            return 0.0;
        double m = mean(v);
        double sum_sq = 0.0;
        for (double x : v)
            sum_sq += (x - m) * (x - m);
        return std::sqrt(sum_sq / (v.size() - 1));
    }
};

/**
 * @brief Run matrix-based solver and collect timing
 */
template <int dim>
TimingResults run_matrix_based_single(const ProblemInterface<dim>& problem,
                                      int n_refinements, int degree,
                                      MPI_Comm comm) {
    SolverParameters params;
    params.verbose = false;
    params.output_solution = false;
    params.max_iterations = 2000;
    params.tolerance = 1e-10;

    MatrixBasedSolver<dim> solver(problem, degree, comm, params);
    solver.run(n_refinements);

    return solver.get_timing_results();
}

/**
 * @brief Run matrix-free solver and collect timing
 */
template <int dim, int fe_degree>
TimingResults run_matrix_free_single(const ProblemInterface<dim>& problem,
                                     int n_refinements, MPI_Comm comm) {
    SolverParameters params;
    params.verbose = false;
    params.output_solution = false;
    params.solver_type = SolverType::MatrixFree;
    params.max_iterations = 2000;
    params.tolerance = 1e-10;
    params.enable_multigrid = true;

    MatrixFreeSolver<dim, fe_degree> solver(problem, comm, params);
    solver.run(n_refinements);

    return solver.get_timing_results();
}

/**
 * @brief Run multiple trials and compute statistics
 */
template <int dim>
ScalingResult run_benchmark_with_trials(const ProblemInterface<dim>& problem,
                                        const std::string& solver_type,
                                        const std::string& test_type,
                                        int n_refinements, int degree,
                                        int n_warmup, int n_trials,
                                        MPI_Comm comm) {
    ScalingResult result;
    result.solver_type = solver_type;
    result.test_type = test_type;
    result.dimension = dim;
    result.n_refinements = n_refinements;
    result.polynomial_degree = degree;

    MPI_Comm_size(comm, &result.n_mpi_processes);
    result.n_threads_per_process = MultithreadInfo::n_threads();
    result.total_cores = result.n_mpi_processes * result.n_threads_per_process;

    std::vector<double> setup_times, assembly_times, solve_times, total_times;
    std::vector<double> iterations, errors, memories;

    // Warmup runs
    for (int w = 0; w < n_warmup; ++w) {
        if (solver_type == "matrix_based") {
            run_matrix_based_single<dim>(problem, n_refinements, degree, comm);
        } else {
            if (degree == 1) {
                run_matrix_free_single<dim, 1>(problem, n_refinements, comm);
            } else if (degree == 2) {
                run_matrix_free_single<dim, 2>(problem, n_refinements, comm);
            } else if (degree == 3) {
                run_matrix_free_single<dim, 3>(problem, n_refinements, comm);
            }
        }
        MPI_Barrier(comm);
    }

    // Timed runs
    for (int t = 0; t < n_trials; ++t) {
        TimingResults timing;

        if (solver_type == "matrix_based") {
            timing = run_matrix_based_single<dim>(problem, n_refinements,
                                                  degree, comm);
        } else {
            if (degree == 1) {
                timing = run_matrix_free_single<dim, 1>(problem, n_refinements,
                                                        comm);
            } else if (degree == 2) {
                timing = run_matrix_free_single<dim, 2>(problem, n_refinements,
                                                        comm);
            } else if (degree == 3) {
                timing = run_matrix_free_single<dim, 3>(problem, n_refinements,
                                                        comm);
            }
        }

        setup_times.push_back(timing.setup_time);
        assembly_times.push_back(timing.assembly_time);
        solve_times.push_back(timing.solve_time);
        total_times.push_back(timing.total_time);
        iterations.push_back(timing.n_iterations);
        errors.push_back(timing.l2_error);
        memories.push_back(timing.memory_mb);

        // Store problem size from first run
        if (t == 0) {
            result.n_dofs = timing.n_dofs;
            result.n_cells = timing.n_cells;
        }

        MPI_Barrier(comm);
    }

    // Compute statistics
    result.setup_time_avg = Statistics::mean(setup_times);
    result.setup_time_std = Statistics::stddev(setup_times);
    result.assembly_time_avg = Statistics::mean(assembly_times);
    result.assembly_time_std = Statistics::stddev(assembly_times);
    result.solve_time_avg = Statistics::mean(solve_times);
    result.solve_time_std = Statistics::stddev(solve_times);
    result.total_time_avg = Statistics::mean(total_times);
    result.total_time_std = Statistics::stddev(total_times);
    result.n_iterations_avg = Statistics::mean(iterations);
    result.l2_error_avg = Statistics::mean(errors);
    result.memory_mb_avg = Statistics::mean(memories);

    // Compute throughput
    result.dofs_per_second = (result.total_time_avg > 0)
                                 ? result.n_dofs / result.total_time_avg
                                 : 0.0;

    return result;
}

/**
 * @brief Strong scaling test for a fixed problem size
 */
template <int dim>
void run_strong_scaling_test(std::vector<ScalingResult>& results,
                             int n_refinements, int degree, int n_warmup,
                             int n_trials, double baseline_time_mb,
                             double baseline_time_mf, MPI_Comm comm) {
    const int rank = Utilities::MPI::this_mpi_process(comm);
    const int n_procs = Utilities::MPI::n_mpi_processes(comm);

    if (rank == 0) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "STRONG SCALING TEST (dim=" << dim << ")\n";
        std::cout << "Fixed refinements: " << n_refinements
                  << ", Degree: " << degree << "\n";
        std::cout << "MPI processes: " << n_procs
                  << ", Threads/process: " << MultithreadInfo::n_threads()
                  << "\n";
        std::cout << std::string(70, '=') << "\n";
    }

    const Problems::ADRProblem<dim> problem;

    // Matrix-based solver
    if (rank == 0)
        std::cout << "\n  Testing Matrix-Based Solver...\n";

    auto mb_result = run_benchmark_with_trials<dim>(
        problem, "matrix_based", "strong_scaling", n_refinements, degree,
        n_warmup, n_trials, comm);

    // Compute scaling metrics
    if (baseline_time_mb > 0) {
        mb_result.speedup = baseline_time_mb / mb_result.total_time_avg;
        mb_result.parallel_efficiency =
            mb_result.speedup / mb_result.total_cores;
    } else {
        mb_result.speedup = 1.0;
        mb_result.parallel_efficiency = 1.0;
    }

    results.push_back(mb_result);

    if (rank == 0) {
        std::cout << "    DoFs: " << mb_result.n_dofs
                  << ", Time: " << mb_result.total_time_avg << "s"
                  << " (+/- " << mb_result.total_time_std << "s)"
                  << ", Speedup: " << mb_result.speedup
                  << ", Efficiency: " << mb_result.parallel_efficiency << "\n";
    }

    // Matrix-free solver
    if (rank == 0)
        std::cout << "  Testing Matrix-Free Solver...\n";

    auto mf_result = run_benchmark_with_trials<dim>(
        problem, "matrix_free", "strong_scaling", n_refinements, degree,
        n_warmup, n_trials, comm);

    if (baseline_time_mf > 0) {
        mf_result.speedup = baseline_time_mf / mf_result.total_time_avg;
        mf_result.parallel_efficiency =
            mf_result.speedup / mf_result.total_cores;
    } else {
        mf_result.speedup = 1.0;
        mf_result.parallel_efficiency = 1.0;
    }

    results.push_back(mf_result);

    if (rank == 0) {
        std::cout << "    DoFs: " << mf_result.n_dofs
                  << ", Time: " << mf_result.total_time_avg << "s"
                  << " (+/- " << mf_result.total_time_std << "s)"
                  << ", Speedup: " << mf_result.speedup
                  << ", Efficiency: " << mf_result.parallel_efficiency << "\n";
        std::cout << "  Matrix-Free vs Matrix-Based Speedup: "
                  << mb_result.total_time_avg / mf_result.total_time_avg
                  << "x\n";
    }
}

/**
 * @brief Weak scaling test - problem size grows with processor count
 */
template <int dim>
void run_weak_scaling_test(std::vector<ScalingResult>& results,
                           int base_refinements, int degree, int n_warmup,
                           int n_trials, double baseline_time_mb,
                           double baseline_time_mf, MPI_Comm comm) {
    const int rank = Utilities::MPI::this_mpi_process(comm);
    const int n_procs = Utilities::MPI::n_mpi_processes(comm);

    // For weak scaling: each doubling of processes should handle 2^dim more
    // cells We add log2(n_procs)/dim refinements to scale problem size
    int extra_refs = 0;
    int temp = n_procs;
    while (temp > 1) {
        extra_refs++;
        temp /= 2;
    }
    const int n_refinements = base_refinements + extra_refs / dim;

    if (rank == 0) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "WEAK SCALING TEST (dim=" << dim << ")\n";
        std::cout << "Base refinements: " << base_refinements
                  << ", Actual: " << n_refinements << "\n";
        std::cout << "MPI processes: " << n_procs
                  << ", Threads/process: " << MultithreadInfo::n_threads()
                  << "\n";
        std::cout << "Target: constant work per core\n";
        std::cout << std::string(70, '=') << "\n";
    }

    const Problems::ADRProblem<dim> problem;

    // Matrix-based solver
    if (rank == 0)
        std::cout << "\n  Testing Matrix-Based Solver...\n";

    auto mb_result = run_benchmark_with_trials<dim>(
        problem, "matrix_based", "weak_scaling", n_refinements, degree,
        n_warmup, n_trials, comm);

    // For weak scaling, efficiency = T(1)/T(p) (ideal = 1.0)
    if (baseline_time_mb > 0) {
        mb_result.parallel_efficiency =
            baseline_time_mb / mb_result.total_time_avg;
        mb_result.speedup = mb_result.parallel_efficiency; // Same for weak
    } else {
        mb_result.parallel_efficiency = 1.0;
        mb_result.speedup = 1.0;
    }

    results.push_back(mb_result);

    if (rank == 0) {
        std::cout << "    DoFs: " << mb_result.n_dofs
                  << ", Time: " << mb_result.total_time_avg << "s"
                  << " (+/- " << mb_result.total_time_std << "s)"
                  << ", Efficiency: " << mb_result.parallel_efficiency << "\n";
    }

    // Matrix-free solver
    if (rank == 0)
        std::cout << "  Testing Matrix-Free Solver...\n";

    auto mf_result = run_benchmark_with_trials<dim>(
        problem, "matrix_free", "weak_scaling", n_refinements, degree, n_warmup,
        n_trials, comm);

    if (baseline_time_mf > 0) {
        mf_result.parallel_efficiency =
            baseline_time_mf / mf_result.total_time_avg;
        mf_result.speedup = mf_result.parallel_efficiency;
    } else {
        mf_result.parallel_efficiency = 1.0;
        mf_result.speedup = 1.0;
    }

    results.push_back(mf_result);

    if (rank == 0) {
        std::cout << "    DoFs: " << mf_result.n_dofs
                  << ", Time: " << mf_result.total_time_avg << "s"
                  << " (+/- " << mf_result.total_time_std << "s)"
                  << ", Efficiency: " << mf_result.parallel_efficiency << "\n";
    }
}

/**
 * @brief Write results to CSV file
 */
void write_results_csv(const std::vector<ScalingResult>& results,
                       const std::string& filename, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank != 0)
        return;

    std::ofstream ofs(filename);
    ofs << ScalingResult::csv_header() << "\n";
    for (const auto& r : results) {
        ofs << r.to_csv() << "\n";
    }
    ofs.close();
    std::cout << "\nResults written to: " << filename << "\n";
}

/**
 * @brief Print summary table
 */
void print_summary(const std::vector<ScalingResult>& results, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank != 0)
        return;

    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "SCALING BENCHMARK SUMMARY\n";
    std::cout << std::string(100, '=') << "\n\n";

    std::cout << std::left << std::setw(12) << "Solver" << std::setw(12)
              << "Test" << std::setw(6) << "Dim" << std::setw(8) << "Cores"
              << std::setw(12) << "DoFs" << std::setw(12) << "Time(s)"
              << std::setw(10) << "Speedup" << std::setw(12) << "Efficiency"
              << std::setw(15) << "DoFs/s"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(12) << r.solver_type
                  << std::setw(12) << r.test_type << std::setw(6) << r.dimension
                  << std::setw(8) << r.total_cores << std::setw(12) << r.n_dofs
                  << std::fixed << std::setprecision(4) << std::setw(12)
                  << r.total_time_avg << std::setprecision(3) << std::setw(10)
                  << r.speedup << std::setw(12) << r.parallel_efficiency
                  << std::scientific << std::setprecision(2) << std::setw(15)
                  << r.dofs_per_second << "\n";
    }
    std::cout << std::string(100, '=') << "\n";
}

/**
 * @brief Print usage information
 */
void print_usage(const char* program_name) {
    std::cout
        << "Usage: mpirun -np <N> " << program_name << " [options]\n\n"
        << "Options:\n"
        << "  --strong          Run strong scaling test only\n"
        << "  --weak            Run weak scaling test only\n"
        << "  --min-ref <n>     Minimum refinements (default: 3)\n"
        << "  --max-ref <n>     Maximum refinements for sweep (default: 7)\n"
        << "  --degree <n>      Polynomial degree 1, 2, or 3 (default: 2)\n"
        << "  --output <file>   Output CSV file prefix (default: "
           "scaling_results)\n"
        << "  --threads <n>     Threads per MPI process (default: auto)\n"
        << "  --trials <n>      Number of timed trials (default: 3)\n"
        << "  --warmup <n>      Number of warmup runs (default: 1)\n"
        << "  --dim <n>         Spatial dimension 2 or 3 (default: 2)\n"
        << "  --help, -h        Show this help message\n";
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
        bool run_strong = true;
        bool run_weak = true;
        int min_refinements = 3;
        int max_refinements = 7;
        int degree = 2;
        int n_threads = -1;
        int n_trials = 3;
        int n_warmup = 1;
        int dimension = 2;
        std::string output_prefix = "scaling_results";

        // Parse command line
        static struct option long_options[] = {
            {"strong", no_argument, nullptr, 's'},
            {"weak", no_argument, nullptr, 'w'},
            {"min-ref", required_argument, nullptr, 'm'},
            {"max-ref", required_argument, nullptr, 'M'},
            {"degree", required_argument, nullptr, 'd'},
            {"output", required_argument, nullptr, 'o'},
            {"threads", required_argument, nullptr, 't'},
            {"trials", required_argument, nullptr, 'n'},
            {"warmup", required_argument, nullptr, 'W'},
            {"dim", required_argument, nullptr, 'D'},
            {"help", no_argument, nullptr, 'h'},
            {nullptr, 0, nullptr, 0}};

        int opt;
        bool explicit_test = false;
        while ((opt = getopt_long(argc, argv, "swm:M:d:o:t:n:W:D:h",
                                  long_options, nullptr)) != -1) {
            switch (opt) {
                case 's':
                    run_strong = true;
                    run_weak = false;
                    explicit_test = true;
                    break;
                case 'w':
                    run_weak = true;
                    run_strong = false;
                    explicit_test = true;
                    break;
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
                    output_prefix = optarg;
                    break;
                case 't':
                    n_threads = std::atoi(optarg);
                    break;
                case 'n':
                    n_trials = std::atoi(optarg);
                    break;
                case 'W':
                    n_warmup = std::atoi(optarg);
                    break;
                case 'D':
                    dimension = std::atoi(optarg);
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

        // If both flags were passed explicitly, run both
        if (!explicit_test) {
            run_strong = true;
            run_weak = true;
        }

        // Set thread limit
        if (n_threads > 0) {
            MultithreadInfo::set_thread_limit(n_threads);
        }

        // Print header
        if (rank == 0) {
            std::cout << "\n" << std::string(70, '*') << "\n";
            std::cout << "  STRONG & WEAK SCALING BENCHMARK\n";
            std::cout << "  Hybrid ADR Solver - Matrix-Based vs Matrix-Free\n";
            std::cout << std::string(70, '*') << "\n\n";
            std::cout << "Configuration:\n";
            std::cout << "  MPI Processes:      " << n_procs << "\n";
            std::cout << "  Threads/Process:    "
                      << MultithreadInfo::n_threads() << "\n";
            std::cout << "  Total Cores:        "
                      << n_procs * MultithreadInfo::n_threads() << "\n";
            std::cout << "  Dimension:          " << dimension << "D\n";
            std::cout << "  Refinement Range:   " << min_refinements << " - "
                      << max_refinements << "\n";
            std::cout << "  Polynomial Degree:  " << degree << "\n";
            std::cout << "  Warmup Runs:        " << n_warmup << "\n";
            std::cout << "  Timed Trials:       " << n_trials << "\n";
            std::cout << "  Run Strong Scaling: " << (run_strong ? "Yes" : "No")
                      << "\n";
            std::cout << "  Run Weak Scaling:   " << (run_weak ? "Yes" : "No")
                      << "\n";
            std::cout << "  Output Prefix:      " << output_prefix << "\n";
        }

        std::vector<ScalingResult> results;

        // Note: baseline times would typically be gathered from a reference run
        // with 1 core. For this single-run benchmark, we set them to 0 to
        // indicate "no baseline" and report efficiency as 1.0 for the first
        // run.
        double baseline_mb = 0.0;
        double baseline_mf = 0.0;

        // Run benchmarks based on dimension
        if (dimension == 2) {
            // Strong scaling: test multiple refinement levels
            if (run_strong) {
                for (int refs = min_refinements; refs <= max_refinements;
                     ++refs) {
                    run_strong_scaling_test<2>(results, refs, degree, n_warmup,
                                               n_trials, baseline_mb,
                                               baseline_mf, comm);
                }
            }

            // Weak scaling
            if (run_weak) {
                for (int base_refs = min_refinements;
                     base_refs <= max_refinements - 2; ++base_refs) {
                    run_weak_scaling_test<2>(results, base_refs, degree,
                                             n_warmup, n_trials, baseline_mb,
                                             baseline_mf, comm);
                }
            }
        } else if (dimension == 3) {
            // 3D: use smaller refinement range due to memory constraints
            int max_ref_3d = std::min(max_refinements, 5);

            if (run_strong) {
                for (int refs = min_refinements; refs <= max_ref_3d; ++refs) {
                    run_strong_scaling_test<3>(results, refs, degree, n_warmup,
                                               n_trials, baseline_mb,
                                               baseline_mf, comm);
                }
            }

            if (run_weak) {
                for (int base_refs = min_refinements;
                     base_refs <= std::max(min_refinements, max_ref_3d - 2);
                     ++base_refs) {
                    run_weak_scaling_test<3>(results, base_refs, degree,
                                             n_warmup, n_trials, baseline_mb,
                                             baseline_mf, comm);
                }
            }
        } else {
            if (rank == 0) {
                std::cerr << "Error: dimension must be 2 or 3\n";
            }
            return 1;
        }

        // Output results
        std::string csv_filename =
            output_prefix + "_" + std::to_string(dimension) + "d_" +
            std::to_string(n_procs) + "mpi_" +
            std::to_string(MultithreadInfo::n_threads()) + "thr.csv";

        write_results_csv(results, csv_filename, comm);
        print_summary(results, comm);

        if (rank == 0) {
            std::cout << "\nBenchmark complete.\n";
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