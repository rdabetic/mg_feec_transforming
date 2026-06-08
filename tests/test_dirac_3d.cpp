#include <gtest/gtest.h>

#include <memory>

#include "dirac_3d.hpp"  // Changed to 3D header
#include "mesh.hpp"
#include "mfem.hpp"
#include "qrsolve.hpp"

using namespace Dirac3D;

// =======================================================================
//                            DIRICHLET TESTS (ESSENTIAL)
// =======================================================================

TEST(DiracCoarseSolver, TestCoarseSolver3DDirichlet_DEC)
{
    mfem::Mesh m =
        mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::TETRAHEDRON, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    // Removed HCURL_2D
    DiracOperator3D op(
        mesh, BCond::Essential, OperatorMode::DEC, MassLumping::RowSum);

    // Assemble the bordered augmented matrix (Size N+1 x N+1)
    auto M = op.assembleDec();

    EigenSparseQRSolver solver(M);

    // 1. Generate a valid exact solution
    mfem::BlockVector sol_exact = op.createBlockVector();
    sol_exact.Randomize(42);
    op.eliminateBC(sol_exact);
    op.eliminateTrivialSolutionHarmonics(sol_exact);

    // 2. Compute the exact RHS (Size N)
    mfem::BlockVector rhs = op.createBlockVector();
    op.Mult(sol_exact, rhs);

    mfem::BlockVector sol(op.getBlockOffsets());
    solver.Mult(rhs, sol);
    op.eliminateTrivialSolutionHarmonics(sol);

    // 6. Verify Results
    mfem::Vector diff = sol_exact;
    diff -= sol;

    EXPECT_NEAR(diff.Norml2() / sol_exact.Norml2(), 0.0, 1e-10);
}

TEST(DiracCoarseSolver, TestCoarseSolver3DDirichlet_FEM)
{
    mfem::Mesh m =
        mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::TETRAHEDRON, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    DiracOperator3D op(
        mesh, BCond::Essential, OperatorMode::Galerkin, MassLumping::RowSum);

    auto M = op.assembleFem();

    EigenSparseQRSolver solver(M);

    mfem::BlockVector sol_exact = op.createBlockVector();
    sol_exact.Randomize(1337);
    op.eliminateBC(sol_exact);
    op.eliminateTrivialSolutionHarmonics(sol_exact);

    mfem::BlockVector rhs = op.createBlockVector();
    op.Mult(sol_exact, rhs);

    mfem::BlockVector sol(op.getBlockOffsets());
    solver.Mult(rhs, sol);
    op.eliminateTrivialSolutionHarmonics(sol);

    mfem::Vector diff = sol_exact;
    diff -= sol;

    mfem::Vector res(sol_exact.Size());
    op.Mult(diff, res);
    EXPECT_NEAR(diff.Norml2() / rhs.Norml2(), 0, 1e-10);
    EXPECT_NEAR(diff.Norml2() / sol_exact.Norml2(), 0.0, 1e-10);
}

// =======================================================================
//                            NATURAL TESTS
// =======================================================================

TEST(DiracCoarseSolver, TestCoarseSolver3DNatural_DEC)
{
    mfem::Mesh m =
        mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::TETRAHEDRON, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    // Natural Boundary Conditions
    DiracOperator3D op(mesh, BCond::Natural, OperatorMode::DEC, MassLumping::RowSum);

    auto M = op.assembleDec();

    EigenSparseQRSolver solver(M);

    mfem::BlockVector sol_exact = op.createBlockVector();
    sol_exact.Randomize(99);
    // No essential DOFs to zero out for Natural BCs
    op.eliminateTrivialSolutionHarmonics(sol_exact);

    mfem::BlockVector rhs = op.createBlockVector();
    op.Mult(sol_exact, rhs);

    mfem::BlockVector sol(op.getBlockOffsets());
    solver.Mult(rhs, sol);
    op.eliminateTrivialSolutionHarmonics(sol);

    mfem::Vector diff = sol_exact;
    diff -= sol;

    EXPECT_NEAR(diff.Norml2() / sol_exact.Norml2(), 0.0, 1e-10);
}

TEST(DiracCoarseSolver, TestCoarseSolver3DNatural_FEM)
{
    mfem::Mesh m =
        mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::TETRAHEDRON, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    DiracOperator3D op(
        mesh, BCond::Natural, OperatorMode::Galerkin, MassLumping::RowSum);

    auto M = op.assembleFem();

    EigenSparseQRSolver solver(M);

    mfem::BlockVector sol_exact = op.createBlockVector();
    sol_exact.Randomize(101);
    op.eliminateTrivialSolutionHarmonics(sol_exact);

    mfem::BlockVector rhs = op.createBlockVector();
    op.applyFemOp(sol_exact, rhs);

    mfem::BlockVector sol(op.getBlockOffsets());
    solver.Mult(rhs, sol);
    op.eliminateTrivialSolutionHarmonics(sol);

    mfem::BlockVector diff = sol_exact;
    diff -= sol;

    EXPECT_NEAR(diff.Norml2() / sol_exact.Norml2(), 0.0, 1e-10);
}

// =======================================================================
//                    SMOOTHER TESTS (NATURAL BC)
// =======================================================================

TEST(DiracSmoother, SmootherConvergence_GaussSeidel_DEC_3D)
{
    mfem::Mesh m =
        mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::TETRAHEDRON, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    // Using Natural BCs as the basic smoother loop doesn't explicitly enforce
    // essential BCs
    auto op = std::make_shared<DiracOperator3D>(
        mesh, BCond::Natural, OperatorMode::DEC, MassLumping::RowSum);

    DiracSmoother3D smoother(op);
    // IMPORTANT: Enable iterative mode so the smoother updates 'x' instead of
    // resetting it to 0
    smoother.iterative_mode = true;

    // 1. Generate exact solution
    mfem::BlockVector sol_exact = op->createBlockVector();
    sol_exact.Randomize(42);
    op->eliminateTrivialSolutionHarmonics(sol_exact);

    // 2. Compute RHS
    mfem::BlockVector rhs = op->createBlockVector();
    op->Mult(sol_exact, rhs);

    // 3. Initialize guess
    mfem::BlockVector x = op->createBlockVector();
    x                   = 0.0;

    mfem::BlockVector diff          = sol_exact;
    mfem::real_t      initial_error = diff.Norml2();

    // 4. Smooth iteratively
    const int max_iter = 50;
    for (int i = 0; i < max_iter; ++i)
        smoother.Mult(rhs, x);

    // 5. Verify Error Reduction
    diff = sol_exact;
    diff -= x;
    mfem::real_t final_error = diff.Norml2();

    // The smoother should have reduced the error by at least an order of
    // magnitude
    EXPECT_LT(final_error, initial_error * 0.1);
}

// =======================================================================
//                    SMOOTHER TESTS (ESSENTIAL BC)
// =======================================================================

TEST(DiracSmoother, SmootherConvergence_GaussSeidel_Essential_DEC_3D)
{
    mfem::Mesh m =
        mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::TETRAHEDRON, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    // Initialize Operator with ESSENTIAL BCs
    auto op = std::make_shared<DiracOperator3D>(
        mesh, BCond::Essential, OperatorMode::DEC, MassLumping::RowSum);

    DiracSmoother3D smoother(op);
    smoother.iterative_mode = true;

    // 1. Generate exact solution and enforce BCs
    mfem::BlockVector sol_exact = op->createBlockVector();
    sol_exact.Randomize(42);
    op->eliminateBC(sol_exact);  // Crucial: Set boundary values to 0
    op->eliminateTrivialSolutionHarmonics(sol_exact);

    // 2. Compute true RHS
    mfem::BlockVector rhs = op->createBlockVector();
    op->Mult(sol_exact, rhs);

    // 3. Initialize guess
    // Setting x = 0.0 satisfies the homogeneous essential BCs from the start
    mfem::BlockVector x = op->createBlockVector();
    x                   = 0.0;

    mfem::BlockVector diff          = sol_exact;
    mfem::real_t      initial_error = diff.Norml2();

    // 4. Smooth iteratively
    const int max_iter = 50;
    for (int i = 0; i < max_iter; ++i)
        smoother.Mult(rhs, x);

    // 5. Verify Error Reduction
    diff = sol_exact;
    diff -= x;
    mfem::real_t final_error = diff.Norml2();

    // Smoother should significantly reduce the error
    EXPECT_LT(final_error, initial_error * 0.1);

    // 6. Verify Strict Essential BC Maintenance
    // If we apply eliminateBC to a copy of x, it sets boundary DOFs to 0.
    // Subtracting the original x leaves ONLY the negated boundary DOFs.
    // The norm of this difference must be zero, meaning the boundary of x was
    // already strictly zero.
    mfem::BlockVector x_bc_check = x;
    op->eliminateBC(x_bc_check);
    x_bc_check -= x;
    EXPECT_NEAR(x_bc_check.Norml2(), 0.0, 1e-12);
}