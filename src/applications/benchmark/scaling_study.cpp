/**
 * @file scaling_study.cpp
 * @brief Fixed scaling benchmark with proper problem sizing
 *
 *
 * Usage:
 *   mpirun -np <N> ./scaling_benchmark [options]
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

// ============================================================================
// Configuration Constants
// ============================================================================

// Minimum DoFs per MPI process for meaningful parallel efficiency
// Below this threshold, communication overhead dominates computation
constexpr unsigned int MIN_DOFS_PER_PROCESS = 50000;

// Target DoFs per process for good scaling (used in weak scaling)
constexpr unsigned int TARGET_DOFS_PER_PROCESS = 100000;

// Estimated DoFs for Q2 elements: (2^refs * degree + 1)^dim
// For 2D Q2: refs=8 -> ~263k, refs=9 -> ~1M, refs=10 -> ~4.2M
// For 3D Q2: refs=4 -> ~35k, refs=5 -> ~275k, refs=6 -> ~2.1M

/**
 * @brief Calculate expected DoFs for given refinement level
 */
template <int dim> unsigned int estimate_dofs(int n_refinements, int degree) {
    unsigned int cells_per_dim = 1u << n_refinements; // 2^refs
    unsigned int dofs_per_dim = cells_per_dim * degree + 1;
    unsigned int total_dofs = 1;
    for (int d = 0; d < dim; ++d) {
        total_dofs *= dofs_per_dim;
    }
    return total_dofs;
}

/**
 * @brief Calculate minimum refinement level for given process count
 */
template <int dim>
int calculate_min_refinements(
    int n_processes, int degree,
    unsigned int min_dofs_per_proc = MIN_DOFS_PER_PROCESS) {
    unsigned int min_total_dofs = n_processes * min_dofs_per_proc;

    for (int refs = 1; refs <= 15; ++refs) {
        if (estimate_dofs<dim>(refs, degree) >= min_total_dofs) {
            return refs;
        }
    }
    return 15; // Maximum reasonable
}

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
    unsigned int dofs_per_process; // NEW: for diagnosing scaling issues

    // Timing (averaged over trials)
    double setup_time_avg;
    double setup_time_std;
    double assembly_time_avg;
    double assembly_time_std;
    double solve_time_avg; // Primary metric for scaling
    double solve_time_std;
    double total_time_avg;
    double total_time_std;

    // Solver stats
    double n_iterations_avg;
    double l2_error_avg;
    double memory_mb_avg;

    // Scaling metrics (computed from SOLVE TIME, not total time)
    double speedup;             // T_solve(1) / T_solve(p)
    double parallel_efficiency; // S(p) / p
    double dofs_per_second;     // Based on solve time

    // Additional metrics
    double speedup_total; // Based on total time (for reference)
    double efficiency_total;

    static std::string csv_header() {
        return "solver_type,test_type,dimension,n_mpi,n_threads,total_cores,"
               "n_refinements,poly_degree,n_dofs,n_cells,dofs_per_process,"
               "setup_time_avg,setup_time_std,"
               "assembly_time_avg,assembly_time_std,"
               "solve_time_avg,solve_time_std,"
               "total_time_avg,total_time_std,"
               "n_iterations_avg,l2_error,memory_mb,"
               "speedup,parallel_efficiency,dofs_per_second,"
               "speedup_total,efficiency_total";
    }

    std::string to_csv() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << solver_type << "," << test_type << "," << dimension << ","
            << n_mpi_processes << "," << n_threads_per_process << ","
            << total_cores << "," << n_refinements << "," << polynomial_degree
            << "," << n_dofs << "," << n_cells << "," << dofs_per_process << ","
            << setup_time_avg << "," << setup_time_std << ","
            << assembly_time_avg << "," << assembly_time_std << ","
            << solve_time_avg << "," << solve_time_std << "," << total_time_avg
            << "," << total_time_std << "," << n_iterations_avg << ","
            << l2_error_avg << "," << memory_mb_avg << "," << speedup << ","
            << parallel_efficiency << "," << dofs_per_second << ","
            << speedup_total << "," << efficiency_total;
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

    // Warmup runs (not timed, allows JIT compilation, cache warming, etc.)
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

        if (t == 0) {
            result.n_dofs = timing.n_dofs;
            result.n_cells = timing.n_cells;
            result.dofs_per_process = timing.n_dofs / result.n_mpi_processes;
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

    // Compute throughput based on SOLVE TIME (not total time)
    result.dofs_per_second = (result.solve_time_avg > 0)
                                 ? result.n_dofs / result.solve_time_avg
                                 : 0.0;

    // Initialize scaling metrics (will be updated later with baseline)
    result.speedup = 1.0;
    result.parallel_efficiency = 1.0;
    result.speedup_total = 1.0;
    result.efficiency_total = 1.0;

    return result;
}

/**
 * @brief Strong scaling test with proper problem sizing
 */
template <int dim>
std::pair<double, double>
run_strong_scaling_test(std::vector<ScalingResult>& results, int n_refinements,
                        int degree, int n_warmup, int n_trials,
                        double baseline_solve_mb, double baseline_solve_mf,
                        double baseline_total_mb, double baseline_total_mf,
                        MPI_Comm comm) {

    const int rank = Utilities::MPI::this_mpi_process(comm);
    const int n_procs = Utilities::MPI::n_mpi_processes(comm);
    const int n_threads = MultithreadInfo::n_threads();
    const int total_cores = n_procs * n_threads;

    // Check if problem is large enough for this core count
    unsigned int estimated_dofs = estimate_dofs<dim>(n_refinements, degree);
    unsigned int dofs_per_proc = estimated_dofs / n_procs;

    if (rank == 0) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "STRONG SCALING TEST (dim=" << dim << ")\n";
        std::cout << "Refinements: " << n_refinements
                  << ", Estimated DoFs: " << estimated_dofs << "\n";
        std::cout << "MPI: " << n_procs << ", Threads: " << n_threads
                  << ", Total Cores: " << total_cores << "\n";
        std::cout << "DoFs per MPI process: " << dofs_per_proc;

        if (dofs_per_proc < MIN_DOFS_PER_PROCESS) {
            std::cout << " (WARNING: below minimum " << MIN_DOFS_PER_PROCESS
                      << ")";
        }
        std::cout << "\n" << std::string(70, '=') << "\n";
    }

    const Problems::ADRProblem<dim> problem;
    double new_baseline_solve_mb = baseline_solve_mb;
    double new_baseline_solve_mf = baseline_solve_mf;

    // Matrix-based solver
    if (rank == 0)
        std::cout << "\n  Testing Matrix-Based Solver...\n";

    auto mb_result = run_benchmark_with_trials<dim>(
        problem, "matrix_based", "strong_scaling", n_refinements, degree,
        n_warmup, n_trials, comm);

    // Compute scaling metrics from SOLVE TIME
    if (baseline_solve_mb > 0) {
        mb_result.speedup = baseline_solve_mb / mb_result.solve_time_avg;
        mb_result.parallel_efficiency = mb_result.speedup / total_cores;
        mb_result.speedup_total = baseline_total_mb / mb_result.total_time_avg;
        mb_result.efficiency_total = mb_result.speedup_total / total_cores;
    } else {
        // This is the baseline run
        new_baseline_solve_mb = mb_result.solve_time_avg;
        mb_result.speedup = 1.0;
        mb_result.parallel_efficiency = 1.0;
        mb_result.speedup_total = 1.0;
        mb_result.efficiency_total = 1.0;
    }

    results.push_back(mb_result);

    if (rank == 0) {
        std::cout << "    DoFs: " << mb_result.n_dofs << " ("
                  << mb_result.dofs_per_process << "/proc)\n";
        std::cout << "    Solve Time: " << mb_result.solve_time_avg << "s"
                  << " (+/- " << mb_result.solve_time_std << "s)\n";
        std::cout << "    Total Time: " << mb_result.total_time_avg << "s\n";
        std::cout << "    Speedup (solve): " << mb_result.speedup
                  << ", Efficiency: " << mb_result.parallel_efficiency * 100
                  << "%\n";
    }

    // Matrix-free solver
    if (rank == 0)
        std::cout << "  Testing Matrix-Free Solver...\n";

    auto mf_result = run_benchmark_with_trials<dim>(
        problem, "matrix_free", "strong_scaling", n_refinements, degree,
        n_warmup, n_trials, comm);

    if (baseline_solve_mf > 0) {
        mf_result.speedup = baseline_solve_mf / mf_result.solve_time_avg;
        mf_result.parallel_efficiency = mf_result.speedup / total_cores;
        mf_result.speedup_total = baseline_total_mf / mf_result.total_time_avg;
        mf_result.efficiency_total = mf_result.speedup_total / total_cores;
    } else {
        new_baseline_solve_mf = mf_result.solve_time_avg;
        mf_result.speedup = 1.0;
        mf_result.parallel_efficiency = 1.0;
        mf_result.speedup_total = 1.0;
        mf_result.efficiency_total = 1.0;
    }

    results.push_back(mf_result);

    if (rank == 0) {
        std::cout << "    DoFs: " << mf_result.n_dofs << " ("
                  << mf_result.dofs_per_process << "/proc)\n";
        std::cout << "    Solve Time: " << mf_result.solve_time_avg << "s"
                  << " (+/- " << mf_result.solve_time_std << "s)\n";
        std::cout << "    Total Time: " << mf_result.total_time_avg << "s\n";
        std::cout << "    Speedup (solve): " << mf_result.speedup
                  << ", Efficiency: " << mf_result.parallel_efficiency * 100
                  << "%\n";
        std::cout << "  MF vs MB solve speedup: "
                  << mb_result.solve_time_avg / mf_result.solve_time_avg
                  << "x\n";
    }

    return {new_baseline_solve_mb, new_baseline_solve_mf};
}

/**
 * @brief Weak scaling test with proper problem scaling
 */
template <int dim>
void run_weak_scaling_test(std::vector<ScalingResult>& results,
                           int base_refinements, int degree, int n_warmup,
                           int n_trials, double baseline_solve_mb,
                           double baseline_solve_mf, MPI_Comm comm) {
    const int rank = Utilities::MPI::this_mpi_process(comm);
    const int n_procs = Utilities::MPI::n_mpi_processes(comm);
    const int n_threads = MultithreadInfo::n_threads();
    const int total_cores = n_procs * n_threads;

    // For weak scaling: scale problem size with processor count
    // Each doubling of processors should increase problem by 2^dim
    // This means adding 1 refinement level per dim doublings of processors
    int extra_refs = 0;
    int temp = n_procs;
    while (temp > 1) {
        extra_refs++;
        temp /= 2;
    }
    const int n_refinements = base_refinements + (extra_refs + dim - 1) / dim;

    unsigned int estimated_dofs = estimate_dofs<dim>(n_refinements, degree);

    if (rank == 0) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "WEAK SCALING TEST (dim=" << dim << ")\n";
        std::cout << "Base refinements: " << base_refinements
                  << ", Actual: " << n_refinements << "\n";
        std::cout << "MPI: " << n_procs << ", Threads: " << n_threads
                  << ", Total Cores: " << total_cores << "\n";
        std::cout << "Estimated DoFs: " << estimated_dofs << " ("
                  << estimated_dofs / n_procs << "/proc)\n";
        std::cout << std::string(70, '=') << "\n";
    }

    const Problems::ADRProblem<dim> problem;

    // Matrix-based solver
    if (rank == 0)
        std::cout << "\n  Testing Matrix-Based Solver...\n";

    auto mb_result = run_benchmark_with_trials<dim>(
        problem, "matrix_based", "weak_scaling", n_refinements, degree,
        n_warmup, n_trials, comm);

    // For weak scaling: efficiency = T(1)/T(p) (ideal = 1.0)
    if (baseline_solve_mb > 0) {
        mb_result.parallel_efficiency =
            baseline_solve_mb / mb_result.solve_time_avg;
        mb_result.speedup = mb_result.parallel_efficiency;
    }

    results.push_back(mb_result);

    if (rank == 0) {
        std::cout << "    DoFs: " << mb_result.n_dofs << " ("
                  << mb_result.dofs_per_process << "/proc)\n";
        std::cout << "    Solve Time: " << mb_result.solve_time_avg << "s\n";
        std::cout << "    Weak Efficiency: "
                  << mb_result.parallel_efficiency * 100 << "%\n";
    }

    // Matrix-free solver
    if (rank == 0)
        std::cout << "  Testing Matrix-Free Solver...\n";

    auto mf_result = run_benchmark_with_trials<dim>(
        problem, "matrix_free", "weak_scaling", n_refinements, degree, n_warmup,
        n_trials, comm);

    if (baseline_solve_mf > 0) {
        mf_result.parallel_efficiency =
            baseline_solve_mf / mf_result.solve_time_avg;
        mf_result.speedup = mf_result.parallel_efficiency;
    }

    results.push_back(mf_result);

    if (rank == 0) {
        std::cout << "    DoFs: " << mf_result.n_dofs << " ("
                  << mf_result.dofs_per_process << "/proc)\n";
        std::cout << "    Solve Time: " << mf_result.solve_time_avg << "s\n";
        std::cout << "    Weak Efficiency: "
                  << mf_result.parallel_efficiency * 100 << "%\n";
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

    std::cout << "\n" << std::string(120, '=') << "\n";
    std::cout << "SCALING BENCHMARK SUMMARY (metrics based on SOLVE TIME)\n";
    std::cout << std::string(120, '=') << "\n\n";

    std::cout << std::left << std::setw(12) << "Solver" << std::setw(10)
              << "Test" << std::setw(6) << "Cores" << std::setw(10) << "DoFs"
              << std::setw(12) << "DoFs/Proc" << std::setw(12) << "Solve(s)"
              << std::setw(12) << "Total(s)" << std::setw(10) << "Speedup"
              << std::setw(12) << "Efficiency" << std::setw(12) << "DoFs/s"
              << "\n";
    std::cout << std::string(120, '-') << "\n";

    for (const auto& r : results) {
        std::string eff_warning =
            (r.dofs_per_process < MIN_DOFS_PER_PROCESS) ? "*" : " ";

        std::cout << std::left << std::setw(12) << r.solver_type
                  << std::setw(10) << r.test_type.substr(0, 8) << std::setw(6)
                  << r.total_cores << std::setw(10) << r.n_dofs << std::setw(12)
                  << std::to_string(r.dofs_per_process) + eff_warning
                  << std::fixed << std::setprecision(4) << std::setw(12)
                  << r.solve_time_avg << std::setw(12) << r.total_time_avg
                  << std::setprecision(2) << std::setw(10) << r.speedup
                  << std::setw(12) << (r.parallel_efficiency * 100)
                  << std::scientific << std::setprecision(2) << std::setw(12)
                  << r.dofs_per_second << "\n";
    }

    std::cout << std::string(120, '=') << "\n";
    std::cout << "* = DoFs/process below recommended minimum ("
              << MIN_DOFS_PER_PROCESS << ")\n";
    std::cout << "Note: Speedup and Efficiency computed from SOLVE TIME only\n";
}

void print_usage(const char* program_name) {
    std::cout
        << "Usage: mpirun -np <N> " << program_name << " [options]\n\n"
        << "Options:\n"
        << "  --strong          Run strong scaling test only\n"
        << "  --weak            Run weak scaling test only\n"
        << "  --min-ref <n>     Minimum refinements (default: "
           "auto-calculated)\n"
        << "  --max-ref <n>     Maximum refinements (default: 10 for 2D, 6 for "
           "3D)\n"
        << "  --degree <n>      Polynomial degree 1, 2, or 3 (default: 2)\n"
        << "  --output <file>   Output CSV file prefix (default: "
           "scaling_results)\n"
        << "  --threads <n>     Threads per MPI process (default: auto)\n"
        << "  --trials <n>      Number of timed trials (default: 3)\n"
        << "  --warmup <n>      Number of warmup runs (default: 1)\n"
        << "  --dim <n>         Spatial dimension 2 or 3 (default: 2)\n"
        << "  --help, -h        Show this help message\n\n"
        << "Notes:\n"
        << "  - Minimum refinement is auto-calculated to ensure at least "
        << MIN_DOFS_PER_PROCESS << " DoFs/process\n"
        << "  - Scaling metrics are computed from SOLVE TIME (not total "
           "time)\n";
}

int main(int argc, char* argv[]) {
    try {
        Utilities::MPI::MPI_InitFinalize mpi_init(
            argc, argv, numbers::invalid_unsigned_int);

        MPI_Comm comm = MPI_COMM_WORLD;
        const int rank = Utilities::MPI::this_mpi_process(comm);
        const int n_procs = Utilities::MPI::n_mpi_processes(comm);

        // Default options
        bool run_strong = true;
        bool run_weak = true;
        int min_refinements = -1; // -1 means auto-calculate
        int max_refinements = -1; // -1 means auto-select based on dimension
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

        if (!explicit_test) {
            run_strong = true;
            run_weak = true;
        }

        if (n_threads > 0) {
            MultithreadInfo::set_thread_limit(n_threads);
        }

        const int actual_threads = MultithreadInfo::n_threads();
        const int total_cores = n_procs * actual_threads;

        // Auto-calculate refinement range based on dimension and core count
        if (dimension == 2) {
            if (min_refinements < 0) {
                min_refinements = calculate_min_refinements<2>(n_procs, degree);
            }
            if (max_refinements < 0) {
                max_refinements = std::max(min_refinements + 2, 10);
            }
        } else {
            if (min_refinements < 0) {
                min_refinements = calculate_min_refinements<3>(n_procs, degree);
            }
            if (max_refinements < 0) {
                max_refinements = std::max(min_refinements + 1, 6);
            }
        }

        // Print header
        if (rank == 0) {
            std::cout << "\n" << std::string(70, '*') << "\n";
            std::cout << "  FIXED SCALING BENCHMARK\n";
            std::cout << "  (Using SOLVE TIME for scaling metrics)\n";
            std::cout << std::string(70, '*') << "\n\n";
            std::cout << "Configuration:\n";
            std::cout << "  MPI Processes:        " << n_procs << "\n";
            std::cout << "  Threads/Process:      " << actual_threads << "\n";
            std::cout << "  Total Cores:          " << total_cores << "\n";
            std::cout << "  Dimension:            " << dimension << "D\n";
            std::cout << "  Refinement Range:     " << min_refinements << " - "
                      << max_refinements << "\n";
            std::cout << "  Polynomial Degree:    " << degree << "\n";
            std::cout << "  Min DoFs/process:     " << MIN_DOFS_PER_PROCESS
                      << "\n";
            std::cout << "  Warmup/Trials:        " << n_warmup << "/"
                      << n_trials << "\n";

            // Show expected problem sizes
            std::cout << "\nExpected problem sizes:\n";
            for (int r = min_refinements; r <= max_refinements; ++r) {
                unsigned int dofs = (dimension == 2)
                                        ? estimate_dofs<2>(r, degree)
                                        : estimate_dofs<3>(r, degree);
                unsigned int dofs_per_proc = dofs / n_procs;
                std::string warning = (dofs_per_proc < MIN_DOFS_PER_PROCESS)
                                          ? " (WARNING: too small!)"
                                          : "";
                std::cout << "  refs=" << r << ": " << dofs << " DoFs ("
                          << dofs_per_proc << "/proc)" << warning << "\n";
            }
        }

        std::vector<ScalingResult> results;

        // Baselines (will be set from first run results)
        double baseline_solve_mb = 0.0, baseline_solve_mf = 0.0;
        double baseline_total_mb = 0.0, baseline_total_mf = 0.0;

        if (dimension == 2) {
            if (run_strong) {
                for (int refs = min_refinements; refs <= max_refinements;
                     ++refs) {
                    auto [new_mb, new_mf] = run_strong_scaling_test<2>(
                        results, refs, degree, n_warmup, n_trials,
                        baseline_solve_mb, baseline_solve_mf, baseline_total_mb,
                        baseline_total_mf, comm);

                    // Update baselines from first run
                    if (baseline_solve_mb == 0.0) {
                        baseline_solve_mb = new_mb;
                        baseline_total_mb = results.back().total_time_avg;
                    }
                    if (baseline_solve_mf == 0.0) {
                        baseline_solve_mf = new_mf;
                        baseline_total_mf = results.back().total_time_avg;
                    }
                }
            }

            if (run_weak) {
                // Reset baselines for weak scaling
                baseline_solve_mb = 0.0;
                baseline_solve_mf = 0.0;

                for (int base_refs = min_refinements;
                     base_refs <= max_refinements - 2; ++base_refs) {
                    run_weak_scaling_test<2>(
                        results, base_refs, degree, n_warmup, n_trials,
                        baseline_solve_mb, baseline_solve_mf, comm);

                    // Get baselines from first weak scaling run
                    if (baseline_solve_mb == 0.0 && results.size() >= 2) {
                        for (auto it = results.rbegin(); it != results.rend();
                             ++it) {
                            if (it->test_type == "weak_scaling") {
                                if (it->solver_type == "matrix_based" &&
                                    baseline_solve_mb == 0.0) {
                                    baseline_solve_mb = it->solve_time_avg;
                                }
                                if (it->solver_type == "matrix_free" &&
                                    baseline_solve_mf == 0.0) {
                                    baseline_solve_mf = it->solve_time_avg;
                                }
                            }
                        }
                    }
                }
            }
        } else if (dimension == 3) {
            if (run_strong) {
                for (int refs = min_refinements; refs <= max_refinements;
                     ++refs) {
                    auto [new_mb, new_mf] = run_strong_scaling_test<3>(
                        results, refs, degree, n_warmup, n_trials,
                        baseline_solve_mb, baseline_solve_mf, baseline_total_mb,
                        baseline_total_mf, comm);

                    if (baseline_solve_mb == 0.0) {
                        baseline_solve_mb = new_mb;
                        baseline_total_mb = results.back().total_time_avg;
                    }
                    if (baseline_solve_mf == 0.0) {
                        baseline_solve_mf = new_mf;
                        baseline_total_mf = results.back().total_time_avg;
                    }
                }
            }

            if (run_weak) {
                baseline_solve_mb = 0.0;
                baseline_solve_mf = 0.0;

                for (int base_refs = min_refinements;
                     base_refs <=
                     std::max(min_refinements, max_refinements - 1);
                     ++base_refs) {
                    run_weak_scaling_test<3>(
                        results, base_refs, degree, n_warmup, n_trials,
                        baseline_solve_mb, baseline_solve_mf, comm);
                }
            }
        } else {
            if (rank == 0)
                std::cerr << "Error: dimension must be 2 or 3\n";
            return 1;
        }

        // Output
        std::string csv_filename = output_prefix + "_" +
                                   std::to_string(dimension) + "d_" +
                                   std::to_string(n_procs) + "mpi_" +
                                   std::to_string(actual_threads) + "thr.csv";

        write_results_csv(results, csv_filename, comm);
        print_summary(results, comm);

        if (rank == 0) {
            std::cout << "\nBenchmark complete.\n";
        }

    } catch (std::exception& exc) {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    }

    return 0;
}