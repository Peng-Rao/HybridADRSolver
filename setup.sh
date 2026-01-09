source /work/u11022931/spack/share/spack/setup-env.sh
spack env create -d .
spack env activate .
spack add llvm@17.0.6
spack add ninja
spack add googletest
spack add dealii+mpi+p4est+petsc~trilinos ^mpich ^petsc
spack install
spack env activate .