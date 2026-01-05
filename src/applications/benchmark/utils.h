/**
 * @file utils.h
 * @brief Utilities for benchmarking hybrid vs fully distributed solvers
 *
 * This header provides:
 * - Timing utilities for accurate performance measurement
 * - Result collection and CSV output
 * - Strong and weak scaling test configurations
 */

#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

namespace BenchmarkUtils {

/**
 * @brief Structure to hold benchmark results for a single run
 */
struct BenchmarkResult {
    // Configuration
    std::string solver_type;   // "matrix_based" or "matrix_free"
    std::string test_type;     // "strong_scaling" or "weak_scaling"
    int n_mpi_processes;       // number of processes
    int n_threads_per_process; // number of threads per process
    int n_refinements;
    int polynomial_degree;

    // Problem size
    unsigned int n_dofs;
    unsigned int n_cells;

    // Timing results (in seconds)
    double setup_time;
    double assembly_time;
    double solve_time;
    double total_time;

    // Solver performance
    unsigned int n_iterations;
    double l2_error;

    // Memory usage (in MB)
    double memory_mb;

    // Derived metrics
    double dofs_per_second;     // Throughput
    double parallel_efficiency; // For scaling analysis

    /**
     * @brief Get CSV header line
     */
    static std::string csv_header() {
        return "solver_type,test_type,n_mpi,n_threads,total_cores,"
               "n_refinements,poly_degree,n_dofs,n_cells,"
               "setup_time,assembly_time,solve_time,total_time,"
               "n_iterations,l2_error,memory_mb,"
               "dofs_per_second,parallel_efficiency";
    }

    /**
     * @brief Convert result to CSV line
     */
    std::string to_csv() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << solver_type << "," << test_type << "," << n_mpi_processes << ","
            << n_threads_per_process << ","
            << (n_mpi_processes * n_threads_per_process) << "," << n_refinements
            << "," << polynomial_degree << "," << n_dofs << "," << n_cells
            << "," << setup_time << "," << assembly_time << "," << solve_time
            << "," << total_time << "," << n_iterations << "," << l2_error
            << "," << memory_mb << "," << dofs_per_second << ","
            << parallel_efficiency;
        return oss.str();
    }
};

/**
 * @brief Class to collect and output benchmark results
 */
class ResultCollector {
public:
    ResultCollector(const std::string& output_file, MPI_Comm comm)
        : filename(output_file), communicator(comm) {
        MPI_Comm_rank(comm, &rank);
    }

    void add_result(const BenchmarkResult& result) {
        results.push_back(result);
    }

    /**
     * @brief Write all results to CSV file (only rank 0)
     */
    void write_csv() const {
        if (rank != 0)
            return;

        std::ofstream ofs(filename);
        ofs << BenchmarkResult::csv_header() << "\n";
        for (const auto& r : results) {
            ofs << r.to_csv() << "\n";
        }
        ofs.close();
    }

    /**
     * @brief Print summary to stdout (only rank 0)
     */
    void print_summary() const {
        if (rank != 0)
            return;

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "BENCHMARK SUMMARY\n";
        std::cout << std::string(80, '=') << "\n\n";

        std::cout << std::left << std::setw(15) << "Solver" << std::setw(10)
                  << "MPI" << std::setw(10) << "Threads" << std::setw(12)
                  << "DoFs" << std::setw(12) << "Total(s)" << std::setw(12)
                  << "Memory(MB)" << std::setw(15) << "DoFs/s"
                  << "\n";
        std::cout << std::string(80, '-') << "\n";

        for (const auto& r : results) {
            std::cout << std::left << std::setw(15) << r.solver_type
                      << std::setw(10) << r.n_mpi_processes << std::setw(10)
                      << r.n_threads_per_process << std::setw(12) << r.n_dofs
                      << std::fixed << std::setprecision(3) << std::setw(12)
                      << r.total_time << std::setw(12) << r.memory_mb
                      << std::scientific << std::setprecision(2)
                      << std::setw(15) << r.dofs_per_second << "\n";
        }
        std::cout << std::string(80, '=') << "\n";
    }

private:
    std::string filename;
    MPI_Comm communicator;
    int rank;
    std::vector<BenchmarkResult> results;
};

/**
 * @brief High-resolution timer using MPI_Wtime for consistency
 */
class Timer {
public:
    void start() { start_time = MPI_Wtime(); }
    void stop() { elapsed = MPI_Wtime() - start_time; }
    double get_elapsed() const { return elapsed; }

private:
    double start_time = 0.0;
    double elapsed = 0.0;
};

/**
 * @brief Compute parallel efficiency
 * @param t1 Time with 1 process (or baseline)
 * @param tp Time with p processes
 * @param p Number of processes
 * @return Parallel efficiency (ideal = 1.0)
 */
inline double compute_efficiency(const double t1, const double tp, int p) {
    if (tp <= 0 || p <= 0)
        return 0.0;
    return t1 / (p * tp);
}

/**
 * @brief Compute speedup
 */
inline double compute_speedup(const double t1, const double tp) {
    if (tp <= 0)
        return 0.0;
    return t1 / tp;
}

/**
 * @brief Get memory info from /proc/self/status (Linux specific)
 */
inline double get_process_memory_mb() {
    double vm_rss = 0.0;
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line.substr(6));
            iss >> vm_rss;    // Value in kB
            vm_rss /= 1024.0; // Convert to MB
            break;
        }
    }
    return vm_rss;
}

/**
 * @brief Configuration for scaling tests
 */
struct ScalingConfig {
    int min_refinements = 3;
    int max_refinements = 6;
    int polynomial_degree = 2;
    int n_warmup_runs = 1;
    int n_timed_runs = 3;
    bool test_matrix_based = true;
    bool test_matrix_free = true;
};

} // namespace BenchmarkUtils

#endif // BENCHMARK_UTILS_H
