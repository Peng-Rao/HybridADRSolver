cmake_minimum_required(VERSION 3.10)

project(HybridADRSolver)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)


find_package(deal.II 9.7.0 REQUIRED HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR
    "\n*** deal.II library not found! ***\n\n"
    )
ENDIF()

# Check for required deal.II features
IF(NOT DEAL_II_WITH_MPI)
  MESSAGE(FATAL_ERROR
    "\n*** deal.II was not configured with MPI support! ***\n\n"
    "This project requires MPI. Please recompile deal.II with:\n"
    "  -DDEAL_II_WITH_MPI=ON\n"
    )
ENDIF()

IF(NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR
    "\n*** deal.II was not configured with p4est support! ***\n\n"
    "This project requires p4est for distributed triangulations.\n"
    "Please recompile deal.II with:\n"
    "  -DDEAL_II_WITH_P4EST=ON\n"
    )
ENDIF()

# add_custom_target(run ALL
#     COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -n 2 $<TARGET_FILE:deadlock>
#     DEPENDS deadlock
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     COMMENT "Launching MPI application with 2 processes"
# )

# deal_ii_initialize_cached_variables()

# add_executable(main main.cc)

# deal_ii_setup_target(main)

# add_executable(mpi_hello main.cpp)
# target_link_libraries(mpi_hello PRIVATE MPI::MPI_CXX)



# add_custom_target(run
#     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe $<TARGET_FILE:mpi_hello>
#     DEPENDS mpi_hello
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     COMMENT "Launching MPI application with 4 processes"
# )


# Print configuration summary
MESSAGE(STATUS "")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "  Hybrid ADR Solver Configuration")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "  deal.II version: ${DEAL_II_PACKAGE_VERSION}")
MESSAGE(STATUS "  MPI support:     ${DEAL_II_WITH_MPI}")
MESSAGE(STATUS "  p4est support:   ${DEAL_II_WITH_P4EST}")
MESSAGE(STATUS "  OpenMP support:  ${OpenMP_CXX_FOUND}")
MESSAGE(STATUS "  OpenMP version:  ${OpenMP_CXX_VERSION}")
MESSAGE(STATUS "  LAPACK support:  ${DEAL_II_WITH_LAPACK}")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "")