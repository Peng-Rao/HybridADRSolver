cmake_minimum_required(VERSION 3.10)

project(HybridADRSolver)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED)


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

# Check TBB support
IF (NOT DEAL_II_WITH_TBB)
    MESSAGE(WARNING
            "\n*** deal.II was not configured with TBB support! ***\n\n"
            "This project can benefit from TBB for multithreading.\n"
            "Consider recompiling deal.II with:\n"
            "  -DDEAL_II_WITH_TBB=ON\n"
    )
ENDIF ()

deal_ii_initialize_cached_variables()

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include/*.cpp")

#ADD_EXECUTABLE(
#        GalerkinSolver
#        src/applications/main.cpp
#)
#
#TARGET_SOURCES(GalerkinSolver PRIVATE ${SRC_FILES})
#
#TARGET_INCLUDE_DIRECTORIES(GalerkinSolver
#        PUBLIC
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
#)
#
#deal_ii_setup_target(GalerkinSolver)

ADD_EXECUTABLE(
        main_matrix_based
        src/applications/main_matrix_based.cpp
)

TARGET_SOURCES(main_matrix_based PRIVATE ${SRC_FILES})

TARGET_INCLUDE_DIRECTORIES(main_matrix_based
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

deal_ii_setup_target(main_matrix_based)

# Print configuration summary
MESSAGE(STATUS "")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "  Hybrid ADR Solver Configuration")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "  deal.II version: ${DEAL_II_PACKAGE_VERSION}")
MESSAGE(STATUS "  MPI support:     ${DEAL_II_WITH_MPI}")
MESSAGE(STATUS "  p4est support:   ${DEAL_II_WITH_P4EST}")
MESSAGE(STATUS "  PETSc support:   ${DEAL_II_WITH_PETSC}")
MESSAGE(STATUS "  TBB support:     ${DEAL_II_WITH_TBB}")
MESSAGE(STATUS "========================================")
MESSAGE(STATUS "")

# =============================================================
# Tests
# =============================================================
# Enable testing
ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)