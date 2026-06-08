cmake_minimum_required(VERSION 3.10)

# ---------------------------------------------------------------------------
# Test Executables
# ---------------------------------------------------------------------------

add_executable(testIncidence
  test_incidence.cpp
)
add_executable(testOperator
  test_op.cpp
)
add_executable(testMG
  test_mg.cpp
)
add_executable(testDirac2D
  test_dirac_2d.cpp
)
add_executable(testDirac3D
  test_dirac_3d.cpp
)
add_executable(testDiracMG
  test_dirac_mg.cpp
)
add_executable(testLaplace
  test_laplace.cpp
)
add_executable(testMagnetostatics
  test_magnetostatics.cpp
)
add_executable(testDiracCVG2D
  test_dirac_2d_cvg.cpp
)
add_executable(testDiracCVG3D
  test_dirac_3d_cvg.cpp
)
add_executable(testLaplaceCVG3D
  test_laplace_3d_cvg.cpp
)
add_executable(testMagnetostaticsCVG3D
  test_magnetostatics_3d_cvg.cpp
)
add_executable(testSpectra
  SpectraOpTest.cpp
)

# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------

target_link_libraries(testIncidence
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  INCIDENCE
  mfem
)

target_link_libraries(testOperator
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  MESH
  mfem
)

target_link_libraries(testDirac2D
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  DIRAC_2D
  mfem
)

target_link_libraries(testDirac3D
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  DIRAC_3D
  mfem
)

target_link_libraries(testDiracMG
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  DIRAC_2D
  DIRAC_3D
  TRANSFER
  mfem
)

target_link_libraries(testLaplace
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  HCURL_LAPLACIAN
  mfem
)

target_link_libraries(testMagnetostatics
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  MAGNETOSTATICS
  mfem
)

target_link_libraries(testDiracCVG2D
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  DIRAC_2D
  mfem
)

target_link_libraries(testDiracCVG3D
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  DIRAC_3D
  mfem
)

target_link_libraries(testLaplaceCVG3D
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  HCURL_LAPLACIAN
  mfem
)

target_link_libraries(testMagnetostaticsCVG3D
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  MAGNETOSTATICS
  mfem
)

target_link_libraries(testMG
  PRIVATE
  GTest::gtest
  GTest::gtest_main
  Eigen3::Eigen
  TRANSFER
  MESH
  mfem
)

target_link_libraries(testSpectra
  PRIVATE
  SPECTRA_INTERFACE
  mfem
  Eigen3::Eigen
  # Prefer imported targets when available (CMake >= 3.5)
  $<$<TARGET_EXISTS:GTest::gtest>:GTest::gtest>
  $<$<TARGET_EXISTS:GTest::gtest_main>:GTest::gtest_main>
  # Fallback for older FindGTest modules
  ${GTEST_LIBRARIES}
  ${GTEST_MAIN_LIBRARIES}
  pthread
)

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

include(CTest)

add_test(NAME Incidence COMMAND testIncidence)
add_test(NAME SpectraInterface COMMAND testSpectra)
add_test(NAME Dirac2D COMMAND testDirac2D)
add_test(NAME Dirac3D COMMAND testDirac3D)
add_test(NAME VectorLaplacian COMMAND testLaplace)
add_test(NAME Magnetostatics COMMAND testMagnetostatics)
add_test(NAME MGClass COMMAND testMG)