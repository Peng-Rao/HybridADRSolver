ADD_EXECUTABLE(test_basic_solver test_basic_solver.cpp)
TARGET_SOURCES(test_basic_solver PRIVATE ${SRC_FILES})
TARGET_INCLUDE_DIRECTORIES(test_basic_solver
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)
deal_ii_setup_target(test_basic_solver)
TARGET_LINK_LIBRARIES(test_basic_solver GTest::GTest GTest::Main)