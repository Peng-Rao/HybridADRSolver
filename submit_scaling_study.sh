#!/bin/bash
#===============================================================================
# submit_scaling_study.sh - Submit multiple PBS jobs for scaling analysis
#===============================================================================
# Cluster Configuration: 5 nodes × 112 cores/node = 560 total cores
# Threading: TBB (deal.II matrix-free) - optimized for fewer ranks, more threads
#
# This script submits a series of jobs with different node/process counts
# to perform a comprehensive scaling study.
#
# Usage: ./submit_scaling_study.sh [--dry-run]
#===============================================================================

set -e

# Cluster Configuration
PROJECT_ID="Hybrid-Parallelism-ADR-Solver-PDE2025"
EMAIL="<YOUR_EMAIL>"
QUEUE="cpu"
WALLTIME="02:00:00"
CPUS_PER_NODE=112      # 112 cores per node
MAX_NODES=5            # 5 nodes available
BUILD_DIR="build"

# Node configurations for scaling study (max 5 nodes)
NODE_COUNTS=(1 2 3 4 5)

# TBB-optimized thread configurations (prefer more threads per rank)
# Format: THREADS_PER_RANK (RANKS_PER_NODE = 112 / THREADS)
# TBB benefits from fewer MPI ranks with more threads for work-stealing
# Options that divide evenly: 112/2=56, 112/4=28, 112/7=16, 112/8=14
THREADS_PER_RANK_OPTIONS=(28 56)  # Corresponding ranks: 4, 2 per node (TBB-optimized)

REFINEMENT_LEVELS=(3 4 5)

# Parse arguments
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - Jobs will not be submitted ==="
fi

# Create job scripts directory
JOB_DIR="pbs_jobs_$(date +%Y%m%d)"
mkdir -p ${JOB_DIR}

echo "==============================================================================="
echo "Scaling Study Job Submission (TBB Threading)"
echo "==============================================================================="
echo "Cluster:        5 nodes × 112 cores/node = 560 total cores"
echo "Project:        ${PROJECT_ID}"
echo "Queue:          ${QUEUE}"
echo "Node counts:    ${NODE_COUNTS[*]}"
echo "Thread configs: ${THREADS_PER_RANK_OPTIONS[*]} (TBB-optimized: more threads/rank)"
echo "Job directory:  ${JOB_DIR}"
echo "==============================================================================="
echo ""

# Function to generate PBS script
generate_pbs_script() {
    local NODES=$1
    local RANKS_PER_NODE=$2
    local THREADS=$3
    local REFINEMENT=$4
    local TEST_TYPE=$5
    
    local TOTAL_RANKS=$((NODES * RANKS_PER_NODE))
    local JOB_NAME="${TEST_TYPE}_n${NODES}_r${RANKS_PER_NODE}_t${THREADS}_ref${REFINEMENT}"
    local SCRIPT_FILE="${JOB_DIR}/${JOB_NAME}.pbs"
    
    cat > ${SCRIPT_FILE} << EOF
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -l select=${NODES}:ncpus=${CPUS_PER_NODE}:mpiprocs=${RANKS_PER_NODE}:ompthreads=${THREADS}
#PBS -l walltime=${WALLTIME}
#PBS -q ${QUEUE}
#PBS -A ${PROJECT_ID}
#PBS -m ae
#PBS -M ${EMAIL}

# Setup logging
cd \${PBS_O_WORKDIR}
LOG_DIR="logs"
mkdir -p \${LOG_DIR}
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
LOG_FILE="\${LOG_DIR}/${JOB_NAME}_\${PBS_JOBID:-local}_\${TIMESTAMP}.log"
exec > >(tee -a "\${LOG_FILE}") 2>&1

# Spack Environment Setup
if [ -f "/work/u11022931/spack/share/spack/setup-env.sh" ]; then
    source /work/u11022931/spack/share/spack/setup-env.sh
elif [ -f "/opt/spack/share/spack/setup-env.sh" ]; then
    source /opt/spack/share/spack/setup-env.sh
else
    echo "ERROR: Spack not found"
    exit 1
fi
spack env activate .

# TBB Threading configuration (primary for deal.II matrix-free)
export TBB_NUM_THREADS=${THREADS}
export OMP_NUM_THREADS=${THREADS}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
ulimit -s unlimited

# Create results subdirectory
RESULTS_DIR="results_scaling/${JOB_NAME}"
mkdir -p \${RESULTS_DIR}

# Auto-detect MPI implementation
MPI_CMD="mpirun"
if \${MPI_CMD} --version 2>&1 | grep -q "Open MPI"; then
    MPI_MAP_OPTS="--map-by ppr:${RANKS_PER_NODE}:node:PE=${THREADS} --bind-to core"
elif \${MPI_CMD} --version 2>&1 | grep -q "HYDRA\|MPICH"; then
    MPI_MAP_OPTS="-ppn ${RANKS_PER_NODE} -bind-to core"
elif \${MPI_CMD} --version 2>&1 | grep -q "Intel"; then
    MPI_MAP_OPTS="-ppn ${RANKS_PER_NODE} -genv I_MPI_PIN=1"
else
    MPI_MAP_OPTS=""
fi

echo "========================================"
echo "Job: ${JOB_NAME}"
echo "Nodes: ${NODES}, Ranks/Node: ${RANKS_PER_NODE}, Threads: ${THREADS}"
echo "Total MPI Ranks: ${TOTAL_RANKS}"
echo "Refinement: ${REFINEMENT}"
echo "Log File: \${LOG_FILE}"
echo "MPI Options: \${MPI_MAP_OPTS}"
echo "========================================"

# Run benchmark
\${MPI_CMD} -np ${TOTAL_RANKS} \${MPI_MAP_OPTS} \\
    ${BUILD_DIR}/benchmark_${TEST_TYPE}_scaling ${REFINEMENT}

# Collect results
mv ${TEST_TYPE}_scaling_*.csv \${RESULTS_DIR}/ 2>/dev/null || true

echo "Job ${JOB_NAME} complete at \$(date)"
echo "Log file: \${LOG_FILE}"
EOF

    echo ${SCRIPT_FILE}
}

# Submit counter
SUBMITTED=0

#-------------------------------------------------------------------------------
# Strong Scaling Jobs
#-------------------------------------------------------------------------------
echo "Generating Strong Scaling Jobs..."
echo "--------------------------------"

for NODES in "${NODE_COUNTS[@]}"; do
    for THREADS in "${THREADS_PER_RANK_OPTIONS[@]}"; do
        # Calculate ranks per node (fill node with MPI+threads)
        RANKS_PER_NODE=$((CPUS_PER_NODE / THREADS))
        
        # Skip invalid configurations
        if [ ${RANKS_PER_NODE} -lt 1 ]; then
            continue
        fi
        
        for REF in "${REFINEMENT_LEVELS[@]}"; do
            SCRIPT=$(generate_pbs_script ${NODES} ${RANKS_PER_NODE} ${THREADS} ${REF} "strong")
            
            echo "  Created: $(basename ${SCRIPT})"
            
            if [ "$DRY_RUN" = false ]; then
                JOB_ID=$(qsub ${SCRIPT})
                echo "    Submitted: ${JOB_ID}"
                ((SUBMITTED++))
            fi
        done
    done
done

#-------------------------------------------------------------------------------
# Weak Scaling Jobs
#-------------------------------------------------------------------------------
echo ""
echo "Generating Weak Scaling Jobs..."
echo "-------------------------------"

# For weak scaling, use TBB-optimized configuration
# Using 28 threads × 4 ranks = 112 cores/node (good for TBB work-stealing)
THREADS=28
RANKS_PER_NODE=$((CPUS_PER_NODE / THREADS))

for NODES in "${NODE_COUNTS[@]}"; do
    for BASE_REF in 2 3; do
        SCRIPT=$(generate_pbs_script ${NODES} ${RANKS_PER_NODE} ${THREADS} ${BASE_REF} "weak")
        
        echo "  Created: $(basename ${SCRIPT})"
        
        if [ "$DRY_RUN" = false ]; then
            JOB_ID=$(qsub ${SCRIPT})
            echo "    Submitted: ${JOB_ID}"
            ((SUBMITTED++))
        fi
    done
done

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------
echo ""
echo "==============================================================================="
echo "Submission Complete"
echo "==============================================================================="

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - No jobs were submitted"
    echo "Job scripts generated in: ${JOB_DIR}/"
    echo "To submit all jobs: for f in ${JOB_DIR}/*.pbs; do qsub \$f; done"
else
    echo "Total jobs submitted: ${SUBMITTED}"
    echo "Monitor with: qstat -u \$USER"
fi

echo ""
echo "Results will be collected in: results_scaling/"
echo "==============================================================================="
