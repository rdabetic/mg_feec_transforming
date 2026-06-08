add_library(INCIDENCE incidence.cpp)
target_link_libraries(INCIDENCE mfem)

add_library(SPECTRA_INTERFACE SpectraErrorOp.cpp)
target_link_libraries(SPECTRA_INTERFACE mfem Eigen3::Eigen)

add_library(TRANSFER transfer.cpp)
target_link_libraries(TRANSFER mfem)

add_library(MESH mesh.cpp)
target_link_libraries(MESH INCIDENCE mfem)

add_library(DIRAC_2D dirac_2d.cpp
    ../inc/mg.tpp ../inc/mg.ipp ../inc/operator.tpp ../inc/operator.ipp)
target_link_libraries(DIRAC_2D mfem MESH TRANSFER)

add_library(DIRAC_3D dirac_3d.cpp
    ../inc/mg.tpp ../inc/mg.ipp ../inc/operator.tpp ../inc/operator.ipp)
target_link_libraries(DIRAC_3D mfem MESH TRANSFER)

add_library(HCURL_LAPLACIAN hcurl_laplacian.cpp block_upper_triangular_preconditioner.cpp
    ../inc/block_upper_triangular_preconditioner.hpp
    ../inc/mg.tpp ../inc/mg.ipp ../inc/operator.tpp ../inc/operator.ipp)
target_link_libraries(HCURL_LAPLACIAN mfem Eigen3::Eigen MESH TRANSFER)

add_library(MAGNETOSTATICS magnetostatics.cpp block_upper_triangular_preconditioner.cpp
    ../inc/block_upper_triangular_preconditioner.hpp
    ../inc/mg.tpp ../inc/mg.ipp ../inc/operator.tpp ../inc/operator.ipp)
target_link_libraries(MAGNETOSTATICS mfem Eigen3::Eigen MESH TRANSFER)

add_library(MULTIGRID
    ../inc/mg.tpp ../inc/mg.ipp ../inc/operator.tpp ../inc/operator.ipp
    ../inc/block_upper_triangular_preconditioner.hpp
    incidence.cpp SpectraErrorOp.cpp mesh.cpp 
    dirac_2d.cpp dirac_3d.cpp hcurl_laplacian.cpp
    magnetostatics.cpp transfer.cpp block_upper_triangular_preconditioner.cpp
)
target_link_libraries(MULTIGRID mfem Eigen3::Eigen)