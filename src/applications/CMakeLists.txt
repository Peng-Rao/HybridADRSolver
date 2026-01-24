set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

ADD_EXECUTABLE(convergence_study convergence_study.cpp)
TARGET_SOURCES(convergence_study PRIVATE ${SRC_FILES})
TARGET_INCLUDE_DIRECTORIES(convergence_study
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)
deal_ii_setup_target(convergence_study)