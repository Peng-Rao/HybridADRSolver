#!/bin/bash
#===============================================================================
# submit_scaling_study.sh - Submit multiple PBS jobs for scaling analysis
#===============================================================================
# This script submits a series of jobs with different node/process counts
# to perform a comprehensive scaling study.
#
# Usage: ./submit_scaling_study.sh [--dry-run]
#===============================================================================

set -e

# Configuration
PROJECT_ID="<YOUR_PROJECT_ID>"
EMAIL="<YOUR_EMAIL>"
QUEUE="normal"
WALLTIME="02:00:00"
CPUS_PER_NODE=32
BUILD_DIR="build"

# Node configurations for scaling study
NODE_COUNTS=(1 2 4 8 16)
THREADS_PER_RANK_OPTIONS=(1 4 8 16)
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
echo "Scaling Study Job Submission"
echo "==============================================================================="
echo "Project:        ${PROJECT_ID}"
echo "Queue:          ${QUEUE}"
echo "Node counts:    ${NODE_COUNTS[*]}"
echo "Thread configs: ${THREADS_PER_RANK_OPTIONS[*]}"
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
#PBS -j oe
#PBS -o ${JOB_DIR}/${JOB_NAME}.log
#PBS -m ae
#PBS -M ${EMAIL}

# Environment setup
cd \${PBS_O_WORKDIR}
module purge
module load gcc/12.2.0
module load openmpi/4.1.4
module load petsc/3.18
module load dealii/9.5.0

# Threading configuration
export OMP_NUM_THREADS=${THREADS}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export TBB_NUM_THREADS=${THREADS}
ulimit -s unlimited

# Create results subdirectory
RESULTS_DIR="results_scaling/${JOB_NAME}"
mkdir -p \${RESULTS_DIR}

echo "========================================"
echo "Job: ${JOB_NAME}"
echo "Nodes: ${NODES}, Ranks/Node: ${RANKS_PER_NODE}, Threads: ${THREADS}"
echo "Total MPI Ranks: ${TOTAL_RANKS}"
echo "Refinement: ${REFINEMENT}"
echo "========================================"

# Run benchmark
mpirun -np ${TOTAL_RANKS} \\
    --map-by ppr:${RANKS_PER_NODE}:node:PE=${THREADS} \\
    --bind-to core \\
    ${BUILD_DIR}/benchmark_${TEST_TYPE}_scaling ${REFINEMENT}

# Collect results
mv ${TEST_TYPE}_scaling_*.csv \${RESULTS_DIR}/ 2>/dev/null || true

echo "Job ${JOB_NAME} complete at \$(date)"
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

# For weak scaling, use fixed threads per rank, vary nodes
THREADS=8
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
