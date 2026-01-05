/**
 * @file weak_scaling.cpp
 * @brief Weak scaling benchmark for hybrid vs distributed solvers
 *
 * Weak scaling: Problem size scales with number of processes.
 * Each process maintains constant work (constant DoFs per process).
 *
 * Ideal weak scaling: Time remains constant as P increases
 * Efficiency = T_1 / T_P (should stay near 1.0)
 * * This tests the communication overhead of both approaches as
 * the system scales up.
 */

#include "core/problem_definition.h"
#include "core/types.h"
#include "matrix_based/matrix_based_solver.h"
#include "matrix_free/matrix_free_solver.h"
#include "utils.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>

#include <fstream>
#include <iomanip>

using namespace dealii;
using namespace HybridADRSolver;
using namespace BenchmarkUtils;

/**
 * @brief Weak scaling test configuration
 */
struct WeakScalingConfig {
    int base_refinements = 3; // Base refinements for single process
    int polynomial_degree = 2;
    int n_warmup_runs = 1;
    int n_timed_runs = 3;
    std::string output_prefix = "weak_scaling";
};

/**
 * @brief Compute refinements for weak scaling
 *
 * For 3D: doubling processes means we can add ~1/3 refinement level
 * to maintain constant DoFs per process.
 *
 * With n_procs processes, we add log2(n_procs)/dim refinement levels.
 */
int compute_weak_scaling_refinements(const int base_refs,
                                     const unsigned int n_procs,
                                     const int dim = 3) {
    if (n_procs <= 1)
        return base_refs;

    // Calculate additional refinements based on process count
    // Each refinement level increases cells by 2^dim
    // So to maintain constant cells/process, we add log_2^dim(n_procs) levels
    double extra = std::log2(static_cast<double>(n_procs)) / dim;
    return base_refs + static_cast<int>(std::round(extra));
}

/**
 * @brief Timing statistics structure
 */
struct TimingStats {
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;

    static TimingStats compute(const std::vector<double>& times) {
        TimingStats stats;
        if (times.empty())
            return stats;

        stats.min = *std::min_element(times.begin(), times.end());
        stats.max = *std::max_element(times.begin(), times.end());

        double sum = 0.0;
        for (const double t : times)
            sum += t;
        stats.mean = sum / times.size();

        double sq_sum = 0.0;
        for (const double t : times)
            sq_sum += (t - stats.mean) * (t - stats.mean);
        stats.stddev = std::sqrt(sq_sum / times.size());

        return stats;
    }
};

/**
 * @brief Benchmark result for weak scaling
 */
struct WeakScalingResult {
    std::string solver_type;
    unsigned int n_mpi_processes{};
    unsigned int n_threads{};
    unsigned int n_refinements{};
    unsigned int n_dofs{};
    unsigned int dofs_per_process{};
    TimingStats timing;
    double memory_mb{};
    double memory_per_process_mb{};
    double efficiency{}; // T_1 / T_P for weak scaling
};

/**
 * @brief Run matrix-based solver benchmark
 */
template <int dim>
WeakScalingResult
run_matrix_based_weak_scaling(const ProblemInterface<dim>& problem,
                              const WeakScalingConfig& config, MPI_Comm comm) {
    auto n_procs = Utilities::MPI::n_mpi_processes(comm);
    auto n_refinements =
        compute_weak_scaling_refinements(config.base_refinements, n_procs, dim);

    std::vector<double> times;
    double memory = 0.0;
    unsigned int n_dofs = 0;

    SolverParameters params;
    params.verbose = false;
    params.output_solution = false;
    params.max_iterations = 2000;
    params.tolerance = 1e-10;

    // Warmup
    for (int i = 0; i < config.n_warmup_runs; ++i) {
        MatrixBasedSolver<dim> solver(problem, config.polynomial_degree, comm,
                                      params);
        solver.run(n_refinements);
    }

    // Timed runs
    for (int i = 0; i < config.n_timed_runs; ++i) {
        double t0 = MPI_Wtime();
        MatrixBasedSolver<dim> solver(problem, config.polynomial_degree, comm,
                                      params);
        solver.run(n_refinements);
        double t1 = MPI_Wtime();
        times.push_back(t1 - t0);

        if (i == 0) {
            const auto& tr = solver.get_timing_results();
            memory = tr.memory_mb;
            n_dofs = tr.n_dofs;
        }
    }

    WeakScalingResult result;
    result.solver_type = "matrix_based";
    result.n_mpi_processes = n_procs;
    result.n_threads = MultithreadInfo::n_threads();
    result.n_refinements = n_refinements;
    result.n_dofs = n_dofs;
    result.dofs_per_process = n_dofs / n_procs;
    result.timing = TimingStats::compute(times);
    result.memory_mb = memory;
    result.memory_per_process_mb = memory / n_procs;

    return result;
}

/**
 * @brief Run matrix-free solver benchmark
 */
template <int dim, int fe_degree>
WeakScalingResult
run_matrix_free_weak_scaling(const ProblemInterface<dim>& problem,
                             const WeakScalingConfig& config, MPI_Comm comm) {
    const int n_procs = Utilities::MPI::n_mpi_processes(comm);
    const int n_refinements =
        compute_weak_scaling_refinements(config.base_refinements, n_procs, dim);

    std::vector<double> times;
    double memory = 0.0;
    unsigned int n_dofs = 0;

    SolverParameters params;
    params.verbose = false;
    params.output_solution = false;
    params.solver_type = SolverType::MatrixFree;
    params.max_iterations = 2000;
    params.tolerance = 1e-10;

    // Warmup
    for (int i = 0; i < config.n_warmup_runs; ++i) {
        MatrixFreeSolver<dim, fe_degree> solver(problem, comm, params);
        solver.run(n_refinements);
    }

    // Timed runs
    for (int i = 0; i < config.n_timed_runs; ++i) {
        double t0 = MPI_Wtime();
        MatrixFreeSolver<dim, fe_degree> solver(problem, comm, params);
        solver.run(n_refinements);
        double t1 = MPI_Wtime();
        times.push_back(t1 - t0);

        if (i == 0) {
            const auto& tr = solver.get_timing_results();
            memory = tr.memory_mb;
            n_dofs = tr.n_dofs;
        }
    }

    WeakScalingResult result;
    result.solver_type = "matrix_free";
    result.n_mpi_processes = n_procs;
    result.n_threads = MultithreadInfo::n_threads();
    result.n_refinements = n_refinements;
    result.n_dofs = n_dofs;
    result.dofs_per_process = n_dofs / n_procs;
    result.timing = TimingStats::compute(times);
    result.memory_mb = memory;
    result.memory_per_process_mb = memory / n_procs;

    return result;
}

void print_result(const WeakScalingResult& r, std::ostream& os) {
    os << std::fixed << std::setprecision(4);
    os << "  " << std::left << std::setw(15) << r.solver_type << std::right
       << std::setw(8) << r.n_mpi_processes << std::setw(8) << r.n_threads
       << std::setw(12) << r.n_dofs << std::setw(12) << r.dofs_per_process
       << std::setw(10) << r.timing.mean << std::setw(12) << r.memory_mb
       << std::setw(12) << r.memory_per_process_mb << "\n";
}

int main(int argc, char* argv[]) {
    try {
        Utilities::MPI::MPI_InitFinalize mpi_init(
            argc, argv, numbers::invalid_unsigned_int);

        MPI_Comm comm = MPI_COMM_WORLD;
        auto rank = Utilities::MPI::this_mpi_process(comm);
        auto n_procs = Utilities::MPI::n_mpi_processes(comm);
        auto n_threads = MultithreadInfo::n_threads();

        ConditionalOStream pcout(std::cout, rank == 0);

        // Configuration
        WeakScalingConfig config;
        config.base_refinements = 3;
        config.polynomial_degree = 2;
        config.n_warmup_runs = 1;
        config.n_timed_runs = 3;

        // Parse command line
        if (argc > 1) {
            config.base_refinements = std::atoi(argv[1]);
        }

        const int actual_refs = compute_weak_scaling_refinements(
            config.base_refinements, n_procs, 3);

        pcout << "\n" << std::string(80, '=') << "\n";
        pcout << "WEAK SCALING BENCHMARK\n";
        pcout << std::string(80, '=') << "\n\n";
        pcout << "Configuration:\n";
        pcout << "  MPI Processes:       " << n_procs << "\n";
        pcout << "  Threads per Rank:    " << n_threads << "\n";
        pcout << "  Total Parallelism:   " << n_procs * n_threads << "\n";
        pcout << "  Base Refinements:    " << config.base_refinements << "\n";
        pcout << "  Actual Refinements:  " << actual_refs << "\n";
        pcout << "  Polynomial Degree:   " << config.polynomial_degree
              << "\n\n";

        pcout << "Note: In weak scaling, problem size increases with process "
                 "count\n";
        pcout << "      to maintain constant work per process. Ideal behavior "
                 "is\n";
        pcout << "      constant execution time as processes increase.\n\n";

        const Problems::ADRProblem<3> problem;

        // Run benchmarks
        pcout << "Running Matrix-Based Solver...\n";
        auto mb_result =
            run_matrix_based_weak_scaling<3>(problem, config, comm);

        pcout << "Running Matrix-Free Solver...\n";
        auto mf_result =
            run_matrix_free_weak_scaling<3, 2>(problem, config, comm);

        // Output results
        pcout << "\n" << std::string(80, '-') << "\n";
        pcout << "RESULTS\n";
        pcout << std::string(80, '-') << "\n\n";

        pcout << "  " << std::left << std::setw(15) << "Solver" << std::right
              << std::setw(8) << "MPI" << std::setw(8) << "Threads"
              << std::setw(12) << "DoFs" << std::setw(12) << "DoFs/Proc"
              << std::setw(10) << "Time(s)" << std::setw(12) << "Memory(MB)"
              << std::setw(12) << "Mem/Proc"
              << "\n";
        pcout << "  " << std::string(77, '-') << "\n";

        print_result(mb_result, std::cout);
        print_result(mf_result, std::cout);

        pcout << "\nComparison:\n";
        pcout << "  Time Ratio (MB/MF):   "
              << mb_result.timing.mean / mf_result.timing.mean << "x\n";
        pcout << "  Memory Ratio (MB/MF): "
              << mb_result.memory_mb / mf_result.memory_mb << "x\n";

        // Write CSV
        if (rank == 0) {
            std::string filename = config.output_prefix + "_" +
                                   std::to_string(n_procs) + "procs.csv";
            std::ofstream ofs(filename);
            ofs << "solver,n_mpi,n_threads,refinements,n_dofs,dofs_per_proc,"
                << "mean_time,min_time,max_time,memory_mb,memory_per_proc_mb\n";
            ofs << std::fixed << std::setprecision(6);

            ofs << mb_result.solver_type << "," << mb_result.n_mpi_processes
                << "," << mb_result.n_threads << "," << mb_result.n_refinements
                << "," << mb_result.n_dofs << "," << mb_result.dofs_per_process
                << "," << mb_result.timing.mean << "," << mb_result.timing.min
                << "," << mb_result.timing.max << "," << mb_result.memory_mb
                << "," << mb_result.memory_per_process_mb << "\n";

            ofs << mf_result.solver_type << "," << mf_result.n_mpi_processes
                << "," << mf_result.n_threads << "," << mf_result.n_refinements
                << "," << mf_result.n_dofs << "," << mf_result.dofs_per_process
                << "," << mf_result.timing.mean << "," << mf_result.timing.min
                << "," << mf_result.timing.max << "," << mf_result.memory_mb
                << "," << mf_result.memory_per_process_mb << "\n";

            ofs.close();
            std::cout << "\nResults written to: " << filename << "\n";
        }

        pcout << std::string(80, '=') << "\n";

    } catch (std::exception& exc) {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    }

    return 0;
}
