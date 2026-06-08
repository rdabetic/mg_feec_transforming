cmake_minimum_required(VERSION 3.28)
project(DECMG LANGUAGES CXX)

# ==============================================================================
# 1. Eigen Strategy: System First, Local Fallback
# ==============================================================================
find_package(Eigen3 QUIET)

if(Eigen3_FOUND)
    message(STATUS "✅ Found System Eigen (Version ${Eigen3_VERSION})")
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
    endif()
else()
    message(STATUS "⚠️ System Eigen not found. Falling back to Local (extern/eigen)")
    set(LOCAL_EIGEN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen")

    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
        target_include_directories(Eigen3::Eigen INTERFACE "${LOCAL_EIGEN_PATH}")
    endif()

    include_directories(BEFORE "${LOCAL_EIGEN_PATH}")
    set(Eigen3_DIR "${LOCAL_EIGEN_PATH}/cmake" CACHE PATH "" FORCE)
    set(EIGEN3_INCLUDE_DIR "${LOCAL_EIGEN_PATH}" CACHE PATH "" FORCE)
    set(Eigen3_FOUND TRUE CACHE BOOL "" FORCE)
endif()

# ==============================================================================
# 2. Standard & Flags
# ==============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp -march=native -mtune=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")

# ==============================================================================
# 3. Spectra
# ==============================================================================
add_library(Spectra INTERFACE)
target_include_directories(Spectra INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/extern/spectra/include")
target_link_libraries(Spectra INTERFACE Eigen3::Eigen)
add_library(Spectra::Spectra ALIAS Spectra)

# ==============================================================================
# 4. Boost (System) - JSON is header-only (Boost >= 1.70)
# ==============================================================================
find_package(Boost REQUIRED)

message(STATUS "✅ Found System Boost: ${Boost_VERSION}")

# ==============================================================================
# 5. MFEM (Submodule)
# ==============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(MFEM_USE_OPENMP ON CACHE BOOL "" FORCE)
endif()

set(MFEM_USE_EIGEN ON CACHE BOOL "Enable Eigen in MFEM" FORCE)

add_subdirectory(extern/mfem)

if(TARGET mfem AND NOT TARGET MFEM::mfem)
    add_library(MFEM::mfem ALIAS mfem)
endif()

# ==============================================================================
# 6. Project Targets
# ==============================================================================
include_directories(
    "${CMAKE_BINARY_DIR}/extern/mfem"
    "${PROJECT_SOURCE_DIR}/inc"
    "/usr/include/suitesparse"
    "${CMAKE_SOURCE_DIR}/extern/spectra/include"
    ${EIGEN3_INCLUDE_DIR}
)
link_directories("${CMAKE_BINARY_DIR}/src")

add_subdirectory(src)
add_subdirectory(targets)

# ==============================================================================
# 7. Tests
# ==============================================================================
set(gtest_force_shared_crt ON CACHE BOOL "Use shared CRT" FORCE)

# Still using the submodule for GoogleTest as requested previously
add_subdirectory(extern/googletest googletest-build)
add_subdirectory(tests)
