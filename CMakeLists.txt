cmake_minimum_required(VERSION 3.10)

project(HybridADRSolver)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)


find_package(deal.II REQUIRED HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
IF (NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR
            "\n*** deal.II library not found! ***\n\n"
    )
ENDIF ()

# Check for required deal.II features
IF (NOT DEAL_II_WITH_MPI)
    MESSAGE(FATAL_ERROR
            "\n*** deal.II was not configured with MPI support! ***\n\n"
            "This project requires MPI. Please recompile deal.II with:\n"
            "  -DDEAL_II_WITH_MPI=ON\n"
    )
ENDIF ()

IF (NOT DEAL_II_WITH_P4EST)
    MESSAGE(FATAL_ERROR
            "\n*** deal.II was not configured with p4est support! ***\n\n"
            "This project requires p4est for distributed triangulations.\n"
            "Please recompile deal.II with:\n"
            "  -DDEAL_II_WITH_P4EST=ON\n"
    )
ENDIF ()

IF (NOT DEAL_II_WITH_PETSC)
    MESSAGE(FATAL_ERROR
            "\n*** deal.II was not configured with PETSc support! ***\n\n"
            "This project requires PETSc for linear algebra backends.\n"
            "Please recompile deal.II with:\n"
            "  -DDEAL_II_WITH_PETSC=ON\n"
    )
ENDIF ()

deal_ii_initialize_cached_variables()

# Enable testing
ENABLE_TESTING()

add_executable(GalerkinSolver src/GalerkinSolver.cc)

deal_ii_setup_target(GalerkinSolver)

add_executable(GalerkinSolverMPI src/GalerkinSolverMPI.cc)
deal_ii_setup_target(GalerkinSolverMPI)

# Print configuration summary
MESSAGE(STATUS "")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "  Hybrid ADR Solver Configuration")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "  deal.II version: ${DEAL_II_PACKAGE_VERSION}")
MESSAGE(STATUS "  MPI support:     ${DEAL_II_WITH_MPI}")
MESSAGE(STATUS "  p4est support:   ${DEAL_II_WITH_P4EST}")
MESSAGE(STATUS "  PETSc support:   ${DEAL_II_WITH_PETSC}")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "")

# =============================================================
# Tests
# =============================================================

# Test: galerkin solver runs and produces output
ADD_TEST(
        NAME test_basic_solver_runs
        COMMAND GalerkinSolver
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Test: Check VTK output is created
ADD_TEST(
        NAME test_output_created
        COMMAND ${CMAKE_COMMAND} -E cat solution-0.vtk
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
SET_TESTS_PROPERTIES(test_output_created PROPERTIES DEPENDS test_basic_solver_runs)