add_executable(cvg cvg.cpp)

find_package(Boost REQUIRED COMPONENTS program_options)

target_link_libraries(cvg mfem MULTIGRID Spectra::Spectra Boost::program_options)
