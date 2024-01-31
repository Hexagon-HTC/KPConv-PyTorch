cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(kpconv_operators LANGUAGES CXX)

include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

find_package(Torch REQUIRED)
find_package(nanoflann REQUIRED)
find_package(OpenMP REQUIRED)

# Define our library target
set(HEADERS
grid_sampling.h
init.h
radius_search.h
)

set(SOURCES
grid_sampling.cpp
init.cpp
radius_search.cpp
)

add_library(kpconv_operators SHARED ${HEADERS} ${SOURCES})

# Enable C++17
target_compile_features(kpconv_operators PRIVATE cxx_range_for)
target_compile_features(kpconv_operators PUBLIC cxx_std_17)

find_package(Threads)

# Link against LibTorch
target_link_libraries(kpconv_operators torch)
target_link_libraries(kpconv_operators nanoflann::nanoflann)
target_link_libraries(kpconv_operators OpenMP::OpenMP_CXX)
target_link_libraries(kpconv_operators Threads::Threads)

if(${CMAKE_VERSION} GREATER_EQUAL "3.18")
    set_property(TARGET kpconv_operators PROPERTY CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86)
endif()

if (MSVC)
    foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
        install(TARGETS kpconv_operators EXPORT kpconv_operators
            ARCHIVE DESTINATION lib/${OUTPUTCONFIG}
            LIBRARY DESTINATION lib/${OUTPUTCONFIG}
            RUNTIME DESTINATION bin/${OUTPUTCONFIG})

        install(FILES $<TARGET_PDB_FILE:kpconv_operators> DESTINATION lib/${OUTPUTCONFIG} OPTIONAL)

        install(EXPORT kpconv_operators DESTINATION lib/cmake/kpconv_operators FILE kpconv_operators-config.cmake)
    endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
else()
    install(TARGETS kpconv_operators EXPORT kpconv_operators
        ARCHIVE DESTINATION lib/kpconv_operators
        LIBRARY DESTINATION lib/kpconv_operators
        RUNTIME DESTINATION bin/kpconv_operators)
    install(EXPORT kpconv_operators DESTINATION lib/cmake/kpconv_operators FILE kpconv_operators-config.cmake)
endif(MSVC)
