#include <gtest/gtest.h>

#include <memory>

#include "mesh.hpp"
#include "mfem.hpp"
#include "qrsolve.hpp"
#include "hcurl_laplacian.hpp"

using namespace HCurlLaplacian;

// =======================================================================
//                            DIRICHLET TESTS (ESSENTIAL)
// =======================================================================

TEST(HCurlLaplacianCoarseSolver, TestCoarseSolver2DDirichlet_DEC)
{
    mfem::Mesh m =
        mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::TRIANGLE, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    HCurlLaplacianOperator op(
        mesh, BCond::Essential, OperatorMode::DEC, MassLumping::RowSum);
    auto M = op.assembleDec();

    EigenSparseQRSolver solver(M);

    // 1. Generate exact solution
    mfem::BlockVector sol_exact = op.createBlockVector();
    sol_exact.Randomize(42);
    op.eliminateBC(sol_exact);  // Enforce u=0 on boundary

    // 2. Compute true RHS
    mfem::BlockVector rhs = op.createBlockVector();
    op.Mult(sol_exact, rhs);
    op.eliminateBC(rhs);  // Zero out boundary RHS

    mfem::BlockVector sol(op.getBlockOffsets());
    solver.Mult(rhs, sol);
    op.eliminateTrivialSolutionHarmonics(sol);

    // 4. Verify Results
    mfem::Vector diff = sol_exact;
    diff -= sol;

    EXPECT_NEAR(diff.Norml2() / sol_exact.Norml2(), 0.0, 1e-10);
}

TEST(HCurlLaplacianCoarseSolver, TestCoarseSolver3DDirichlet_FEM)
{
    mfem::Mesh m =
        mfem::Mesh::MakeCartesian3D(3, 3, 3, mfem::Element::TETRAHEDRON, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    HCurlLaplacianOperator op(
        mesh, BCond::Essential, OperatorMode::Galerkin, MassLumping::RowSum);
    auto M = op.assembleFem();

    EigenSparseQRSolver solver(M);

    mfem::BlockVector sol_exact = op.createBlockVector();
    sol_exact.Randomize(1337);
    op.eliminateBC(sol_exact);

    mfem::BlockVector rhs = op.createBlockVector();
    op.Mult(sol_exact, rhs);
    op.eliminateBC(rhs);

    mfem::BlockVector sol(op.getBlockOffsets());
    solver.Mult(rhs, sol);
    op.eliminateTrivialSolutionHarmonics(sol);

    mfem::Vector diff = sol_exact;
    diff -= sol;

    EXPECT_NEAR(diff.Norml2() / sol_exact.Norml2(), 0.0, 1e-10);
}

// =======================================================================
//                            NATURAL TESTS
// =======================================================================

TEST(HCurlLaplacianCoarseSolver, TestCoarseSolver2DNatural_DEC)
{
    mfem::Mesh m =
        mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::TRIANGLE, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    HCurlLaplacianOperator op(
        mesh, BCond::Natural, OperatorMode::DEC, MassLumping::RowSum);
    auto M = op.assembleDec();

    EigenSparseQRSolver solver(M);

    // 1. Generate exact solution
    mfem::BlockVector sol_exact = op.createBlockVector();
    sol_exact.Randomize(99);
    op.eliminateTrivialSolutionHarmonics(
        sol_exact);  // Remove constant nullspace

    // 2. Compute true RHS
    mfem::BlockVector rhs = op.createBlockVector();
    op.Mult(sol_exact, rhs);

    mfem::BlockVector sol(op.getBlockOffsets());
    solver.Mult(rhs, sol);
    op.eliminateTrivialSolutionHarmonics(sol);

    mfem::Vector diff = sol_exact;
    diff -= sol;

    EXPECT_NEAR(diff.Norml2() / sol_exact.Norml2(), 0.0, 1e-10);
}

// =======================================================================
//                    SMOOTHER TESTS (NATURAL BC)
// =======================================================================

TEST(HCurlLaplacianSmoother, SmootherConvergence_Natural_DEC_2D)
{
    mfem::Mesh m =
        mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::TRIANGLE, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    auto op = std::make_shared<HCurlLaplacianOperator>(
        mesh, BCond::Natural, OperatorMode::DEC, MassLumping::RowSum);

    HCurlLaplacianSmoother smoother(op);
    smoother.iterative_mode =
        true;  // Crucial for continuous V-cycle smoothing updates

    mfem::BlockVector sol_exact = op->createBlockVector();
    sol_exact.Randomize(42);
    op->eliminateTrivialSolutionHarmonics(sol_exact);

    mfem::BlockVector rhs = op->createBlockVector();
    op->Mult(sol_exact, rhs);

    mfem::BlockVector x = op->createBlockVector();
    x                   = 0.0;

    mfem::BlockVector diff          = sol_exact;
    mfem::real_t      initial_error = diff.Norml2();

    const int max_iter = 40;
    for (int i = 0; i < max_iter; ++i)
        smoother.Mult(rhs, x);

    diff = sol_exact;
    diff -= x;
    mfem::real_t final_error = diff.Norml2();

    // Distributive relaxation should rapidly slash the error
    EXPECT_LT(final_error, initial_error * 0.33);
}

// =======================================================================
//                    SMOOTHER TESTS (ESSENTIAL BC)
// =======================================================================

TEST(HCurlLaplacianSmoother, SmootherConvergence_Essential_DEC_3D)
{
    mfem::Mesh m =
        mfem::Mesh::MakeCartesian3D(3, 3, 3, mfem::Element::TETRAHEDRON, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    auto op = std::make_shared<HCurlLaplacianOperator>(
        mesh, BCond::Essential, OperatorMode::DEC, MassLumping::RowSum);

    HCurlLaplacianSmoother smoother(op);
    smoother.iterative_mode = true;

    mfem::BlockVector sol_exact = op->createBlockVector();
    sol_exact.Randomize(123);
    op->eliminateBC(sol_exact);  // Strict boundary enforcement
    op->eliminateTrivialSolutionHarmonics(sol_exact);

    mfem::BlockVector rhs = op->createBlockVector();
    op->Mult(sol_exact, rhs);
    op->eliminateBC(rhs);  // Zero rhs boundaries

    mfem::BlockVector x = op->createBlockVector();
    x                   = 0.0;

    mfem::BlockVector diff          = sol_exact;
    mfem::real_t      initial_error = diff.Norml2();

    const int max_iter = 40;
    for (int i = 0; i < max_iter; ++i)
        smoother.Mult(rhs, x);

    diff = sol_exact;
    diff -= x;
    mfem::real_t final_error = diff.Norml2();

    EXPECT_LT(final_error, initial_error * 0.1);

    // Verify Strict Essential BC Maintenance
    mfem::BlockVector x_bc_check = x;
    op->eliminateBC(x_bc_check);
    x_bc_check -= x;
    EXPECT_NEAR(x_bc_check.Norml2(), 0.0, 1e-12);
}
