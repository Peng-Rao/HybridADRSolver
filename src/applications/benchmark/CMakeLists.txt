set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

ADD_EXECUTABLE(scaling_study scaling_study.cpp)
TARGET_SOURCES(scaling_study PRIVATE ${SRC_FILES})
TARGET_INCLUDE_DIRECTORIES(scaling_study
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    target_compile_options(scaling_study PRIVATE -march=native)
endif ()
deal_ii_setup_target(scaling_study)

ADD_EXECUTABLE(convergence_study convergence_study.cpp)
TARGET_SOURCES(convergence_study PRIVATE ${SRC_FILES})
TARGET_INCLUDE_DIRECTORIES(convergence_study
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)
deal_ii_setup_target(convergence_study)

ADD_EXECUTABLE(main_2d main_2d.cpp)
TARGET_SOURCES(main_2d PRIVATE ${SRC_FILES})
TARGET_INCLUDE_DIRECTORIES(main_2d
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)
deal_ii_setup_target(main_2d)