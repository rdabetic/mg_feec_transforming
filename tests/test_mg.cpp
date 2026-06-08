#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include "mesh.hpp"
#include "mfem.hpp"
#include "mg.tpp"
#include "operator.tpp"

// =======================================================================
//                            DUMMY CLASSES
// =======================================================================

// A dummy smoother passing through the Mult
class DummySmoother : public mfem::Solver
{
public:
    DummySmoother(std::shared_ptr<mfem::Operator> op) : mfem::Solver(0)
    {
        iterative_mode = true;
    }

    void SetOperator(const mfem::Operator& op) override
    {
        height = op.Height();
        width  = op.Width();
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override
    {
        // Mock smoothing behavior: just copy x to y for dummy purposes
        y = x;
    }
};

// A concrete implementation of the WhitneyFormOperator
template <unsigned int... FORMS>
class DummyOp : public WhitneyFormOperator<DummyOp<FORMS...>, FORMS...>
{
public:
    // Inherit the constructor so Multigrid can extract configurations and
    // instantiate it
    using WhitneyFormOperator<DummyOp<FORMS...>, FORMS...>::WhitneyFormOperator;

    void applyDecOp(const mfem::BlockVector& in, mfem::BlockVector& out)
        const override
    {
    }
    void applyFemOp(const mfem::BlockVector& in, mfem::BlockVector& out)
        const override
    {
    }
    // void smoother(mfem::BlockVector& u, const mfem::BlockVector& rhs) const
    // override {}

    // Return a dummy identity matrix to bypass UMFPack solver checks at level 0
    std::shared_ptr<mfem::SparseMatrix> assembleFem() override
    {
        // Create a basic identity matrix based on the operator's height
        int  size = std::max(1, this->Height());  // ensure non-zero for dummy
        auto mat  = std::make_shared<mfem::SparseMatrix>(size);
        for (int i = 0; i < size; ++i)
            mat->Add(i, i, 1.0);
        mat->Finalize();
        return mat;
    }

    std::shared_ptr<mfem::SparseMatrix> assembleDec() override
    {
        return assembleFem();
    }
};

// =======================================================================
//                            TEST CASES
// =======================================================================

class MultigridTest : public ::testing::Test
{
protected:
    std::shared_ptr<Mesh> base_mesh;

    void SetUp() override
    {
        // Create a base 2D Cartesian Quadrilateral Mesh
        mfem::Mesh m =
            mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::TRIANGLE, true);
        base_mesh = std::make_shared<Mesh>(std::move(m));
    }
};

TEST_F(MultigridTest, AutomatedUniformRefinementAndGetters)
{
    // 1. Create Level 0 operator mapping 0-forms (Nodes) and 1-forms (Edges)
    auto coarse_op = std::make_shared<DummyOp<0, 1>>(
        base_mesh, BCond::Essential, OperatorMode::DEC, MassLumping::RowSum);

    // 2. Instantiate the Multigrid Solver with the derived type
    Multigrid<DummyOp<0, 1>, DummySmoother, 0, 1> mg;

    // Test empty hierarchy getters
    EXPECT_THROW(mg.getFinestOperator(), std::logic_error);

    // 3. Add Coarse Level
    mg.addLevel(coarse_op);
    EXPECT_EQ(mg.numLevels(), 1);

    auto fetched_coarse = mg.getFinestOperator();
    EXPECT_EQ(fetched_coarse, coarse_op);

    const int dofs_level_0 = coarse_op->Height();

    // 4. Trigger automated uniform refinement
    EXPECT_NO_THROW(mg.addRefinedLevel());
    EXPECT_EQ(mg.numLevels(), 2);

    // 5. Test new getters on the refined level
    auto finest_op = mg.getFinestOperator();
    EXPECT_NE(finest_op, nullptr);

    // Verify it is strictly a different instance than the coarse operator
    EXPECT_NE(finest_op, coarse_op);

    // 6. Verify Refinement configuration extraction
    EXPECT_EQ(finest_op->getMode(), coarse_op->getMode());
    EXPECT_EQ(finest_op->getMassLumping(), coarse_op->getMassLumping());

    // A uniform refinement of a 2x2 quad mesh yields a 4x4 mesh.
    // Thus the overall operator Height should increase.
    const int dofs_level_1 = finest_op->Height();
    EXPECT_GT(dofs_level_1, dofs_level_0);
    EXPECT_EQ(mg.Height(), dofs_level_1);
}
