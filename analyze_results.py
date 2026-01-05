#!/usr/bin/env python3
"""
analyze_results.py - Analyze and visualize benchmark results

This script generates:
1. Strong scaling plots (speedup and efficiency)
2. Weak scaling plots (time vs processes)
3. Memory efficiency comparison
4. Throughput analysis (DoFs/second)
5. Summary tables

Usage:
    python analyze_results.py --results-dir results/ --output-dir figures/
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11

# Color scheme
COLORS = {
    'matrix_based': '#1f77b4',  # Blue
    'matrix_free': '#ff7f0e',   # Orange
    'ideal': '#2ca02c',         # Green (dashed)
}

MARKERS = {
    'matrix_based': 'o',
    'matrix_free': 's',
}

LABELS = {
    'matrix_based': 'Matrix-Based (MPI Distributed)',
    'matrix_free': 'Matrix-Free (Hybrid MPI+Threading)',
}


def load_results(results_dir):
    """Load all CSV results from directory."""
    all_data = []

    # Find all CSV files
    csv_files = glob.glob(os.path.join(results_dir, '**/*.csv'), recursive=True)

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df['source_file'] = os.path.basename(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")

    if not all_data:
        print("No data files found!")
        return None

    # Combine all dataframes
    combined = pd.concat(all_data, ignore_index=True)

    # Compute derived metrics if not present
    if 'total_cores' not in combined.columns:
        combined['total_cores'] = combined['n_mpi'] * combined.get('n_threads', 1)

    if 'dofs_per_second' not in combined.columns and 'n_dofs' in combined.columns:
        time_col = 'mean_time' if 'mean_time' in combined.columns else 'total_time'
        combined['dofs_per_second'] = combined['n_dofs'] / combined[time_col]

    return combined


def plot_strong_scaling(df, output_dir):
    """Generate strong scaling plots."""
    # Filter for strong scaling data
    strong_df = df[df['source_file'].str.contains('strong', case=False, na=False)]

    if strong_df.empty:
        print("No strong scaling data found")
        return

    time_col = 'mean_time' if 'mean_time' in strong_df.columns else 'total_time'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Execution time vs cores
    ax1 = axes[0]
    for solver in ['matrix_based', 'matrix_free']:
        solver_df = strong_df[strong_df['solver'] == solver]
        if not solver_df.empty:
            solver_df = solver_df.sort_values('total_cores')
            ax1.plot(solver_df['total_cores'], solver_df[time_col],
                     marker=MARKERS.get(solver, 'o'),
                     color=COLORS.get(solver, 'gray'),
                     label=LABELS.get(solver, solver),
                     linewidth=2, markersize=8)

    ax1.set_xlabel('Number of Cores')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Strong Scaling: Execution Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')

    # Plot 2: Speedup
    ax2 = axes[1]

    # Calculate speedup relative to single-core baseline
    for solver in ['matrix_based', 'matrix_free']:
        solver_df = strong_df[strong_df['solver'] == solver].sort_values('total_cores')
        if not solver_df.empty and len(solver_df) > 0:
            t1 = solver_df[solver_df['total_cores'] == solver_df['total_cores'].min()][time_col].values[0]
            speedup = t1 / solver_df[time_col]
            ax2.plot(solver_df['total_cores'], speedup,
                     marker=MARKERS.get(solver, 'o'),
                     color=COLORS.get(solver, 'gray'),
                     label=LABELS.get(solver, solver),
                     linewidth=2, markersize=8)

    # Ideal speedup line
    cores = sorted(strong_df['total_cores'].unique())
    ax2.plot(cores, cores, '--', color=COLORS['ideal'],
             label='Ideal Speedup', linewidth=2)

    ax2.set_xlabel('Number of Cores')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Strong Scaling: Speedup')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strong_scaling.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'strong_scaling.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Strong scaling plots saved to {output_dir}")


def plot_weak_scaling(df, output_dir):
    """Generate weak scaling plots."""
    weak_df = df[df['source_file'].str.contains('weak', case=False, na=False)]

    if weak_df.empty:
        print("No weak scaling data found")
        return

    time_col = 'mean_time' if 'mean_time' in weak_df.columns else 'total_time'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Execution time (should be constant for ideal weak scaling)
    ax1 = axes[0]
    for solver in ['matrix_based', 'matrix_free']:
        solver_df = weak_df[weak_df['solver'] == solver].sort_values('n_mpi')
        if not solver_df.empty:
            ax1.plot(solver_df['n_mpi'], solver_df[time_col],
                     marker=MARKERS.get(solver, 'o'),
                     color=COLORS.get(solver, 'gray'),
                     label=LABELS.get(solver, solver),
                     linewidth=2, markersize=8)

    ax1.set_xlabel('Number of MPI Processes')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Weak Scaling: Execution Time\n(Ideal: constant time)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)

    # Plot 2: Parallel efficiency
    ax2 = axes[1]
    for solver in ['matrix_based', 'matrix_free']:
        solver_df = weak_df[weak_df['solver'] == solver].sort_values('n_mpi')
        if not solver_df.empty and len(solver_df) > 0:
            t1 = solver_df[solver_df['n_mpi'] == solver_df['n_mpi'].min()][time_col].values[0]
            efficiency = t1 / solver_df[time_col]
            ax2.plot(solver_df['n_mpi'], efficiency,
                     marker=MARKERS.get(solver, 'o'),
                     color=COLORS.get(solver, 'gray'),
                     label=LABELS.get(solver, solver),
                     linewidth=2, markersize=8)

    # Ideal efficiency line
    ax2.axhline(y=1.0, color=COLORS['ideal'], linestyle='--',
                label='Ideal Efficiency', linewidth=2)

    ax2.set_xlabel('Number of MPI Processes')
    ax2.set_ylabel('Parallel Efficiency')
    ax2.set_title('Weak Scaling: Efficiency\n(Ideal: 100%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weak_scaling.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'weak_scaling.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Weak scaling plots saved to {output_dir}")


def plot_memory_comparison(df, output_dir):
    """Generate memory efficiency comparison plots."""
    # Get data with memory info
    mem_df = df[df['memory_mb'].notna()].copy()

    if mem_df.empty:
        print("No memory data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Memory vs DoFs
    ax1 = axes[0]
    for solver in ['matrix_based', 'matrix_free']:
        solver_df = mem_df[mem_df['solver'] == solver].sort_values('n_dofs')
        if not solver_df.empty:
            ax1.plot(solver_df['n_dofs'], solver_df['memory_mb'],
                     marker=MARKERS.get(solver, 'o'),
                     color=COLORS.get(solver, 'gray'),
                     label=LABELS.get(solver, solver),
                     linewidth=2, markersize=8)

    ax1.set_xlabel('Number of DoFs')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Memory Efficiency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot 2: Memory ratio (matrix-based / matrix-free)
    ax2 = axes[1]

    # Group by DoFs and compare
    grouped = mem_df.groupby(['n_dofs', 'solver'])['memory_mb'].mean().unstack()
    if 'matrix_based' in grouped.columns and 'matrix_free' in grouped.columns:
        ratio = grouped['matrix_based'] / grouped['matrix_free']
        ratio = ratio.dropna()
        ax2.bar(range(len(ratio)), ratio.values, color=COLORS['matrix_based'], alpha=0.7)
        ax2.set_xticks(range(len(ratio)))
        ax2.set_xticklabels([f'{int(d):,}' for d in ratio.index], rotation=45)
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2)

    ax2.set_xlabel('Number of DoFs')
    ax2.set_ylabel('Memory Ratio (Matrix-Based / Matrix-Free)')
    ax2.set_title('Memory Savings with Matrix-Free\n(>1 means matrix-free uses less)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_comparison.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'memory_comparison.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Memory comparison plots saved to {output_dir}")


def plot_throughput(df, output_dir):
    """Generate throughput (DoFs/second) comparison."""
    if 'dofs_per_second' not in df.columns:
        print("No throughput data found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for solver in ['matrix_based', 'matrix_free']:
        solver_df = df[df['solver'] == solver].sort_values('total_cores')
        if not solver_df.empty:
            ax.plot(solver_df['total_cores'], solver_df['dofs_per_second'],
                    marker=MARKERS.get(solver, 'o'),
                    color=COLORS.get(solver, 'gray'),
                    label=LABELS.get(solver, solver),
                    linewidth=2, markersize=8)

    ax.set_xlabel('Number of Cores')
    ax.set_ylabel('Throughput (DoFs/second)')
    ax.set_title('Solver Throughput Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'throughput.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Throughput plot saved to {output_dir}")


def generate_summary_table(df, output_dir):
    """Generate summary statistics table."""
    time_col = 'mean_time' if 'mean_time' in df.columns else 'total_time'

    summary = df.groupby('solver').agg({
        time_col: ['mean', 'min', 'max'],
        'memory_mb': ['mean', 'min', 'max'],
        'n_dofs': ['mean', 'max'],
    }).round(3)

    summary_file = os.path.join(output_dir, 'summary_stats.csv')
    summary.to_csv(summary_file)

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(summary.to_string())
    print("\nSaved to:", summary_file)


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--results-dir', default='results',
                        help='Directory containing CSV results')
    parser.add_argument('--output-dir', default='figures',
                        help='Directory for output figures')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading results from: {args.results_dir}")
    df = load_results(args.results_dir)

    if df is None or df.empty:
        print("No data to analyze!")
        return

    print(f"Loaded {len(df)} data points")
    print(f"Solvers: {df['solver'].unique()}")

    # Generate plots
    plot_strong_scaling(df, args.output_dir)
    plot_weak_scaling(df, args.output_dir)
    plot_memory_comparison(df, args.output_dir)
    plot_throughput(df, args.output_dir)
    generate_summary_table(df, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
