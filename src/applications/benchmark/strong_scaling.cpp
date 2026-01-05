/**
 * @file strong_scaling.cpp
 * @brief Strong scaling benchmark for hybrid vs distributed solvers
 *
 * Strong scaling: Fixed problem size, varying number of processes/threads.
 * This measures how well the solvers parallelize a fixed workload.
 *
 * Ideal strong scaling: Time = T_1 / P (where P = number of processes)
 * Efficiency = T_1 / (P * T_P)
 *
 * For hybrid parallelization, total parallelism = MPI_ranks × threads_per_rank
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
 * @brief Strong scaling test configuration
 */
struct StrongScalingConfig {
    int n_refinements = 4; // Fixed problem size
    int polynomial_degree = 2;
    int n_warmup_runs = 1;
    int n_timed_runs = 3; // Average over multiple runs
    std::string output_prefix = "strong_scaling";
};

/**
 * @brief Run matrix-based solver and collect timing
 */
template <int dim>
std::vector<double> benchmark_matrix_based(const ProblemInterface<dim>& problem,
                                           const StrongScalingConfig& config,
                                           MPI_Comm comm) {
    std::vector<double> times;

    SolverParameters params;
    params.verbose = false;
    params.output_solution = false;
    params.max_iterations = 2000;
    params.tolerance = 1e-10;

    // Warmup runs
    for (int i = 0; i < config.n_warmup_runs; ++i) {
        MatrixBasedSolver<dim> solver(problem, config.polynomial_degree, comm,
                                      params);
        solver.run(config.n_refinements);
    }

    // Timed runs
    for (int i = 0; i < config.n_timed_runs; ++i) {
        const double t0 = MPI_Wtime();
        MatrixBasedSolver<dim> solver(problem, config.polynomial_degree, comm,
                                      params);
        solver.run(config.n_refinements);
        const double t1 = MPI_Wtime();
        times.push_back(t1 - t0);
    }

    return times;
}

/**
 * @brief Run matrix-free solver and collect timing
 */
template <int dim, int fe_degree>
std::vector<double> benchmark_matrix_free(const ProblemInterface<dim>& problem,
                                          const StrongScalingConfig& config,
                                          MPI_Comm comm) {
    std::vector<double> times;

    SolverParameters params;
    params.verbose = false;
    params.output_solution = false;
    params.solver_type = SolverType::MatrixFree;
    params.max_iterations = 2000;
    params.tolerance = 1e-10;

    // Warmup runs
    for (int i = 0; i < config.n_warmup_runs; ++i) {
        MatrixFreeSolver<dim, fe_degree> solver(problem, comm, params);
        solver.run(config.n_refinements);
    }

    // Timed runs
    for (int i = 0; i < config.n_timed_runs; ++i) {
        const double t0 = MPI_Wtime();
        MatrixFreeSolver<dim, fe_degree> solver(problem, comm, params);
        solver.run(config.n_refinements);
        const double t1 = MPI_Wtime();
        times.push_back(t1 - t0);
    }

    return times;
}

/**
 * @brief Compute statistics from timing data
 */
struct TimingStats {
    double mean;
    double min;
    double max;
    double stddev;

    static TimingStats compute(const std::vector<double>& times) {
        TimingStats stats = {};
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

int main(int argc, char* argv[]) {
    try {
        // Initialize MPI with full threading support
        Utilities::MPI::MPI_InitFinalize mpi_init(
            argc, argv, numbers::invalid_unsigned_int);

        MPI_Comm comm = MPI_COMM_WORLD;
        auto rank = Utilities::MPI::this_mpi_process(comm);
        auto n_procs = Utilities::MPI::n_mpi_processes(comm);
        auto n_threads = MultithreadInfo::n_threads();

        ConditionalOStream pcout(std::cout, rank == 0);

        // Configuration
        StrongScalingConfig config;
        config.n_refinements = 4; // Adjust based on available memory
        config.polynomial_degree = 2;
        config.n_warmup_runs = 1;
        config.n_timed_runs = 3;

        // Parse command line for refinements
        if (argc > 1) {
            config.n_refinements = std::atoi(argv[1]);
        }

        pcout << "\n" << std::string(70, '=') << "\n";
        pcout << "STRONG SCALING BENCHMARK\n";
        pcout << std::string(70, '=') << "\n\n";
        pcout << "Configuration:\n";
        pcout << "  MPI Processes:      " << n_procs << "\n";
        pcout << "  Threads per Rank:   " << n_threads << "\n";
        pcout << "  Total Parallelism:  " << n_procs * n_threads << "\n";
        pcout << "  Mesh Refinements:   " << config.n_refinements << "\n";
        pcout << "  Polynomial Degree:  " << config.polynomial_degree << "\n";
        pcout << "  Warmup Runs:        " << config.n_warmup_runs << "\n";
        pcout << "  Timed Runs:         " << config.n_timed_runs << "\n\n";

        const Problems::ADRProblem<3> problem;

        // Run matrix-based benchmark
        pcout << "Running Matrix-Based Solver (MPI distributed)...\n";
        auto mb_times = benchmark_matrix_based<3>(problem, config, comm);
        auto mb_stats = TimingStats::compute(mb_times);

        // Run matrix-free benchmark
        pcout << "Running Matrix-Free Solver (Hybrid MPI+threading)...\n";
        auto mf_times = benchmark_matrix_free<3, 2>(problem, config, comm);
        auto mf_stats = TimingStats::compute(mf_times);

        // Output results
        pcout << "\n" << std::string(70, '-') << "\n";
        pcout << "RESULTS (time in seconds)\n";
        pcout << std::string(70, '-') << "\n\n";

        pcout << std::fixed << std::setprecision(4);
        pcout << "Matrix-Based Solver:\n";
        pcout << "  Mean:   " << mb_stats.mean << " s\n";
        pcout << "  Min:    " << mb_stats.min << " s\n";
        pcout << "  Max:    " << mb_stats.max << " s\n";
        pcout << "  StdDev: " << mb_stats.stddev << " s\n\n";

        pcout << "Matrix-Free Solver (Hybrid):\n";
        pcout << "  Mean:   " << mf_stats.mean << " s\n";
        pcout << "  Min:    " << mf_stats.min << " s\n";
        pcout << "  Max:    " << mf_stats.max << " s\n";
        pcout << "  StdDev: " << mf_stats.stddev << " s\n\n";

        pcout << "Comparison:\n";
        pcout << "  Speedup (MB/MF): " << mb_stats.mean / mf_stats.mean
              << "x\n";

        // Write results to CSV
        if (rank == 0) {
            std::string filename = config.output_prefix + "_" +
                                   std::to_string(n_procs) + "mpi_" +
                                   std::to_string(n_threads) + "threads.csv";
            std::ofstream ofs(filename);
            ofs << "solver,n_mpi,n_threads,refinements,mean_time,min_time,max_"
                   "time,stddev\n";
            ofs << std::fixed << std::setprecision(6);
            ofs << "matrix_based," << n_procs << "," << n_threads << ","
                << config.n_refinements << "," << mb_stats.mean << ","
                << mb_stats.min << "," << mb_stats.max << "," << mb_stats.stddev
                << "\n";
            ofs << "matrix_free," << n_procs << "," << n_threads << ","
                << config.n_refinements << "," << mf_stats.mean << ","
                << mf_stats.min << "," << mf_stats.max << "," << mf_stats.stddev
                << "\n";
            ofs.close();

            std::cout << "\nResults written to: " << filename << "\n";
        }

        pcout << std::string(70, '=') << "\n";

    } catch (std::exception& exc) {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    }

    return 0;
}
