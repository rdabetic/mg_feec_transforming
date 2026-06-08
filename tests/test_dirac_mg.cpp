#include <gtest/gtest.h>

#include <memory>

#include "dirac_2d.hpp"
#include "dirac_3d.hpp"
#include "mesh.hpp"
#include "mfem.hpp"
#include "mg.tpp"

// =======================================================================
//                            2D MULTIGRID TESTS
// =======================================================================

TEST(MultigridSolver, VCycle2D_Essential_DEC)
{
    using namespace Dirac2D;

    // 1. Create Coarse Mesh
    mfem::Mesh m_coarse =
        mfem::Mesh::MakeCartesian2D(5, 6, mfem::Element::TRIANGLE, true);
    auto coarse_mesh = std::make_shared<Mesh>(std::move(m_coarse));

    // 2. Instantiate Coarse Operator
    auto coarse_op = std::make_shared<DiracOperator2D>(
        coarse_mesh, BCond::Essential, OperatorMode::DEC, MassLumping::RowSum);

    // 3. Setup Multigrid Hierarchy
    Multigrid<DiracOperator2D, DiracSmoother2D, 0, 1, 2> mg(OperatorMode::DEC);
    mg.setCycleType(CycleType::VCYCLE);
    mg.setSmoothingIterations(1, 1);
    mg.setIterativeMode(true);

    // Add Level 0 (Coarsest, automatically triggers direct coarse solver)
    mg.addLevel(coarse_op);

    // Add Level 1 (Finer, uses Gauss-Seidel Smoother)
    mg.addRefinedLevel();

    // Retrieve finest operator. It comes back as the base WhitneyFormOperator,
    // so we downcast it to access Dirac-specific harmonic elimination.
    auto finest_op = std::dynamic_pointer_cast<const DiracOperator2D>(
        mg.getFinestOperator());
    ASSERT_NE(finest_op, nullptr)
        << "Failed to retrieve and downcast the finest 2D operator.";

    // 4. Generate exact solution on the finest level
    mfem::BlockVector sol_exact = finest_op->createBlockVector();
    sol_exact.Randomize(42);
    finest_op->eliminateBC(sol_exact);  // Enforce u=0 on boundary
    finest_op->eliminateTrivialSolutionHarmonics(
        sol_exact);  // Orthogonalize against constants

    // 5. Compute RHS
    mfem::BlockVector rhs = finest_op->createBlockVector();
    finest_op->Mult(sol_exact, rhs);
    finest_op->eliminateBC(rhs);  // Crucial for essential BC tracking

    // 6. Initialize guess
    mfem::BlockVector x = finest_op->createBlockVector();
    x                   = 0.0;

    mfem::BlockVector diff          = sol_exact;
    mfem::real_t      initial_error = diff.Norml2();

    // 7. Iterate the Multigrid V-Cycle
    const int max_iter = 16;
    for (int i = 0; i < max_iter; ++i)
        mg.Mult(rhs, x);

    // 8. Verify Convergence
    diff = sol_exact;
    diff -= x;
    mfem::real_t final_error = diff.Norml2();

    // V-cycles are highly efficient; 15 iterations should easily slash the
    // error
    EXPECT_LT(final_error, initial_error * 0.05);
}

// =======================================================================
//                            3D MULTIGRID TESTS
// =======================================================================

TEST(MultigridSolver, VCycle3D_Essential_DEC)
{
    using namespace Dirac3D;

    // 1. Create Coarse Mesh
    mfem::Mesh m_coarse =
        mfem::Mesh::MakeCartesian3D(3, 4, 5, mfem::Element::TETRAHEDRON, true);
    auto coarse_mesh = std::make_shared<Mesh>(std::move(m_coarse));

    // 2. Instantiate Coarse Operator (No HCURL_2D param for 3D)
    auto coarse_op = std::make_shared<DiracOperator3D>(
        coarse_mesh, BCond::Essential, OperatorMode::DEC, MassLumping::RowSum);

    // 3. Setup Multigrid Hierarchy
    Multigrid<DiracOperator3D, DiracSmoother3D, 0, 1, 2, 3> mg(
        OperatorMode::DEC);
    mg.setCycleType(CycleType::VCYCLE);
    mg.setSmoothingIterations(1, 1);
    mg.setIterativeMode(true);

    mg.addLevel(coarse_op);
    mg.addRefinedLevel();

    auto finest_op = std::dynamic_pointer_cast<const DiracOperator3D>(
        mg.getFinestOperator());
    ASSERT_NE(finest_op, nullptr)
        << "Failed to retrieve and downcast the finest 3D operator.";

    // 4. Generate exact solution
    mfem::BlockVector sol_exact = finest_op->createBlockVector();
    sol_exact.Randomize(1337);
    finest_op->eliminateBC(sol_exact);
    finest_op->eliminateTrivialSolutionHarmonics(sol_exact);

    // 5. Compute RHS
    mfem::BlockVector rhs = finest_op->createBlockVector();
    finest_op->Mult(sol_exact, rhs);
    finest_op->eliminateBC(rhs);

    // 6. Initialize guess
    mfem::BlockVector x = finest_op->createBlockVector();
    x                   = 0.0;

    mfem::BlockVector diff          = sol_exact;
    mfem::real_t      initial_error = diff.Norml2();

    // 7. Iterate the Multigrid V-Cycle
    const int max_iter = 15;
    for (int i = 0; i < max_iter; ++i)
        mg.Mult(rhs, x);

    // 8. Verify Convergence
    diff = sol_exact;
    diff -= x;
    mfem::real_t final_error = diff.Norml2();

    EXPECT_LT(final_error, initial_error * 0.05);
}

// =======================================================================
//                    2D MULTIGRID TESTS (NATURAL BC)
// =======================================================================

TEST(MultigridSolver, VCycle2D_Natural_DEC)
{
    using namespace Dirac2D;

    // 1. Create Coarse Mesh
    mfem::Mesh m_coarse =
        mfem::Mesh::MakeCartesian2D(5, 6, mfem::Element::TRIANGLE, true);
    auto coarse_mesh = std::make_shared<Mesh>(std::move(m_coarse));

    // 2. Instantiate Coarse Operator with NATURAL BC
    auto coarse_op = std::make_shared<DiracOperator2D>(
        coarse_mesh, BCond::Natural, OperatorMode::DEC, MassLumping::RowSum);

    // 3. Setup Multigrid Hierarchy
    Multigrid<DiracOperator2D, DiracSmoother2D, 0, 1, 2> mg(OperatorMode::DEC);
    mg.setCycleType(CycleType::VCYCLE);
    mg.setSmoothingIterations(1, 1);
    mg.setIterativeMode(true);

    mg.addLevel(coarse_op);
    mg.addRefinedLevel();

    auto finest_op = std::dynamic_pointer_cast<const DiracOperator2D>(
        mg.getFinestOperator());
    ASSERT_NE(finest_op, nullptr)
        << "Failed to retrieve and downcast the finest 2D operator.";
    ASSERT_EQ(finest_op->getBCond(), BCond::Natural) << "Wrong BC!";

    // 4. Generate exact solution on the finest level
    mfem::BlockVector sol_exact = finest_op->createBlockVector();
    sol_exact.Randomize(42);
    finest_op->eliminateTrivialSolutionHarmonics(sol_exact); 

    // 5. Compute RHS
    mfem::BlockVector rhs = finest_op->createBlockVector();
    finest_op->Mult(sol_exact, rhs);

    // 6. Initialize guess
    mfem::BlockVector x = finest_op->createBlockVector();
    x                   = 0.0;

    mfem::BlockVector diff          = sol_exact;
    mfem::real_t      initial_error = diff.Norml2();

    // 7. Iterate the Multigrid V-Cycle
    const int max_iter = 16;
    for (int i = 0; i < max_iter; ++i)
        mg.Mult(rhs, x);

    // 8. Verify Convergence
    diff = sol_exact;
    diff -= x;
    mfem::real_t final_error = diff.Norml2();

    EXPECT_LT(final_error, initial_error * 0.05);
}

// =======================================================================
//                    3D MULTIGRID TESTS (NATURAL BC)
// =======================================================================

TEST(MultigridSolver, VCycle3D_Natural_DEC)
{
    using namespace Dirac3D;

    // 1. Create Coarse Mesh
    mfem::Mesh m_coarse =
        mfem::Mesh::MakeCartesian3D(3, 4, 5, mfem::Element::TETRAHEDRON, true);
    auto coarse_mesh = std::make_shared<Mesh>(std::move(m_coarse));

    // 2. Instantiate Coarse Operator with NATURAL BC
    auto coarse_op = std::make_shared<DiracOperator3D>(
        coarse_mesh, BCond::Natural, OperatorMode::DEC, MassLumping::RowSum);

    // 3. Setup Multigrid Hierarchy
    Multigrid<DiracOperator3D, DiracSmoother3D, 0, 1, 2, 3> mg(
        OperatorMode::DEC);
    mg.setCycleType(CycleType::VCYCLE);
    mg.setSmoothingIterations(1, 1);
    mg.setIterativeMode(true);

    mg.addLevel(coarse_op);
    mg.addRefinedLevel();  

    auto finest_op = std::dynamic_pointer_cast<const DiracOperator3D>(
        mg.getFinestOperator());
    ASSERT_NE(finest_op, nullptr)
        << "Failed to retrieve and downcast the finest 3D operator.";

    // 4. Generate exact solution
    mfem::BlockVector sol_exact = finest_op->createBlockVector();
    sol_exact.Randomize(1337);
    finest_op->eliminateTrivialSolutionHarmonics(sol_exact);

    // 5. Compute RHS
    mfem::BlockVector rhs = finest_op->createBlockVector();
    finest_op->Mult(sol_exact, rhs);

    // 6. Initialize guess
    mfem::BlockVector x = finest_op->createBlockVector();
    x                   = 0.0;

    mfem::BlockVector diff          = sol_exact;
    mfem::real_t      initial_error = diff.Norml2();

    // 7. Iterate the Multigrid V-Cycle
    const int max_iter = 15;
    for (int i = 0; i < max_iter; ++i)
        mg.Mult(rhs, x);

    // 8. Verify Convergence
    diff = sol_exact;
    diff -= x;
    mfem::real_t final_error = diff.Norml2();

    EXPECT_LT(final_error, initial_error * 0.05);
}
