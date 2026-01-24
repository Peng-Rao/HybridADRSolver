set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

ADD_EXECUTABLE(main_matrix_based main_matrix_based.cpp)
TARGET_SOURCES(main_matrix_based PRIVATE ${SRC_FILES})
TARGET_INCLUDE_DIRECTORIES(main_matrix_based
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)
deal_ii_setup_target(main_matrix_based)

ADD_EXECUTABLE(main_matrix_free main_matrix_free.cpp)
TARGET_SOURCES(main_matrix_free PRIVATE ${SRC_FILES})
TARGET_INCLUDE_DIRECTORIES(main_matrix_free
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)
deal_ii_setup_target(main_matrix_free)