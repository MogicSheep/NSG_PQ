set(CMAKE_CXX_STANDARD 11)

add_executable(test_nsg_index test_nsg_index.cpp test_product_quantization_search.cpp test_distance_calculation.cpp distance_test.cpp test_hybrid_product.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME})

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME})

add_executable(test_product_quantization_search test_product_quantization_search.cpp)
target_link_libraries(test_product_quantization_search ${PROJECT_NAME})

add_executable(test_distance_calculation test_distance_calculation.cpp)
target_link_libraries(test_distance_calculation ${PROJECT_NAME})

add_executable(distance_test distance_test.cpp)
target_link_libraries(distance_test ${PROJECT_NAME})

add_executable(test_hybrid_product test_hybrid_product.cpp)
target_link_libraries(test_hybrid_product ${PROJECT_NAME})