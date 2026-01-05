set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

ADD_EXECUTABLE(benchmark_strong_scaling strong_scaling.cpp)
TARGET_SOURCES(benchmark_strong_scaling PRIVATE ${SRC_FILES})
TARGET_INCLUDE_DIRECTORIES(benchmark_strong_scaling
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)
deal_ii_setup_target(benchmark_strong_scaling)

ADD_EXECUTABLE(benchmark_weak_scaling weak_scaling.cpp)
TARGET_SOURCES(benchmark_weak_scaling PRIVATE ${SRC_FILES})
TARGET_INCLUDE_DIRECTORIES(benchmark_weak_scaling
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)
deal_ii_setup_target(benchmark_weak_scaling)

ADD_EXECUTABLE(benchmark_main benchmark_main.cpp)
TARGET_SOURCES(benchmark_main PRIVATE ${SRC_FILES})
TARGET_INCLUDE_DIRECTORIES(benchmark_main
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)
deal_ii_setup_target(benchmark_main)