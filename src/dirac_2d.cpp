#include "dirac_2d.hpp"

using namespace Dirac2D;

void DiracOperator2D::applyDecOp(
    const mfem::BlockVector& in,
    mfem::BlockVector&       out) const
{
    getExteriorDerivative(0).Mult(in.GetBlock(0), out.GetBlock(1));
    getExteriorDerivative(1).Mult(in.GetBlock(1), out.GetBlock(2));
    getCodifferential(1).Mult(in.GetBlock(1), out.GetBlock(0));
    getCodifferential(2).AddMult(in.GetBlock(2), out.GetBlock(1));
}

void DiracOperator2D::applyFemOp(
    const mfem::BlockVector& in,
    mfem::BlockVector&       out) const
{
    // 0-forms
    getMassMatrix(1)->Mult(in.GetBlock(1), out.GetBlock(1));
    getExteriorDerivative(0).MultTranspose(out.GetBlock(1), out.GetBlock(0));
    // 1-forms
    getMassMatrix(2)->Mult(in.GetBlock(2), out.GetBlock(2));
    getExteriorDerivative(1).MultTranspose(out.GetBlock(2), out.GetBlock(1));

    mfem::Vector tmp(out.GetBlock(1).Size());
    getExteriorDerivative(0).Mult(in.GetBlock(0), tmp);
    getMassMatrix(1)->AddMult(tmp, out.GetBlock(1));
    // 2-forms
    tmp.SetSize(out.GetBlock(2).Size());
    getExteriorDerivative(1).Mult(in.GetBlock(1), tmp);
    getMassMatrix(2)->Mult(tmp, out.GetBlock(2));
}

std::shared_ptr<mfem::SparseMatrix> DiracOperator2D::assembleDec()
{
    auto block_mat = std::make_unique<mfem::BlockMatrix>(this->block_offsets_);

    block_mat->SetBlock(0, 1, delta_[1].get());
    block_mat->SetBlock(1, 0, d_[0].get());
    block_mat->SetBlock(1, 2, delta_[2].get());
    block_mat->SetBlock(2, 1, d_[1].get());

    // --- ENFORCE BCond::Essential BCs ---
    // If BC is BCond::Essential, we must place 1.0 on the diagonal of boundary
    // rows to ensure the system is non-singular and satisfies u=0 on the
    // boundary.
    std::unique_ptr<mfem::SparseMatrix> diag0, diag1;
    if (bc_ == BCond::Essential)
    {
        // 0-form diagonal block (0,0)
        diag0 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(0));
        const mfem::Array<int>& ess_dofs0 = *this->getEssentialTrueDofs(0);
        for (int i = 0; i < ess_dofs0.Size(); ++i)
            diag0->Set(ess_dofs0[i], ess_dofs0[i], 1.0);
        diag0->Finalize();
        block_mat->SetBlock(0, 0, diag0.get());

        // 1-form diagonal block (1,1)
        diag1 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(1));
        const mfem::Array<int>& ess_dofs1 = *this->getEssentialTrueDofs(1);
        for (int i = 0; i < ess_dofs1.Size(); ++i)
            diag1->Set(ess_dofs1[i], ess_dofs1[i], 1.0);
        diag1->Finalize();
        block_mat->SetBlock(1, 1, diag1.get());
    }

    auto M = std::shared_ptr<mfem::SparseMatrix>(block_mat->CreateMonolithic());
    M->Finalize();

    return M;
}

std::shared_ptr<mfem::SparseMatrix> DiracOperator2D::assembleFem()
{
    auto bD0 = std::make_unique<mfem::MixedBilinearForm>(
        this->spaces_[0].get(), this->spaces_[1].get());
    auto bD1 = std::make_unique<mfem::MixedBilinearForm>(
        this->spaces_[1].get(), this->spaces_[2].get());

    mfem::ConstantCoefficient one(1.0);
    bD0->AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator(one));
    bD1->AddDomainIntegrator(new mfem::MixedScalarCurlIntegrator(one));

    bD0->Assemble();
    bD0->Finalize();
    bD1->Assemble();
    bD1->Finalize();

    auto D0 = std::unique_ptr<mfem::SparseMatrix>(bD0->LoseMat());
    auto D1 = std::unique_ptr<mfem::SparseMatrix>(bD1->LoseMat());

    // --- ENFORCE BCond::Essential BCs ---
    if (bc_ == BCond::Essential)
    {
        // 1. Eliminate entries in the off-diagonal blocks

        // For Vertex DOFs (0-forms):
        // Monolithic Row 0 depends on D0^T. We zero Row i of D0^T by zeroing
        // Col i of D0.
        D0->EliminateCols(*this->essential_dof_masks_[0]);

        // For Edge DOFs (1-forms):
        // Monolithic Row 1 depends on D0 and D1^T.
        // Zero Row j of D0:
        for (auto k : *this->essential_dof_[1])
            D0->EliminateRow(k);
        // Zero Row j of D1^T by zeroing Col j of D1:
        D1->EliminateCols(*this->essential_dof_masks_[1]);
    }

    auto D0T = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*D0));
    auto D1T = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*D1));

    auto block_mat = std::make_unique<mfem::BlockMatrix>(this->block_offsets_);

    block_mat->SetBlock(0, 1, D0T.get());
    block_mat->SetBlock(1, 0, D0.get());
    block_mat->SetBlock(1, 2, D1T.get());
    block_mat->SetBlock(2, 1, D1.get());

    // --- ENFORCE BCond::Essential BCs ---
    // If BC is BCond::Essential, we must place 1.0 on the diagonal of boundary
    // rows to ensure the system is non-singular and satisfies u=0 on the
    // boundary.
    std::unique_ptr<mfem::SparseMatrix> diag0, diag1;
    if (bc_ == BCond::Essential)
    {
        // Diagonal for 0-forms
        diag0 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(0));
        for (int i : *this->essential_dof_[0])
            diag0->Set(i, i, 1.0);
        diag0->Finalize();
        block_mat->SetBlock(0, 0, diag0.get());

        // Diagonal for 1-forms
        diag1 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(1));
        for (int i : *this->essential_dof_[1])
            diag1->Set(i, i, 1.0);
        diag1->Finalize();
        block_mat->SetBlock(1, 1, diag1.get());
    }

    auto M = std::shared_ptr<mfem::SparseMatrix>(block_mat->CreateMonolithic());
    M->Finalize();

    return M;
}

// -----------------------------------------------------------------------
// DiracSmoother2D Implementation
// -----------------------------------------------------------------------

DiracSmoother2D::DiracSmoother2D(std::shared_ptr<const DiracOperator2D> op)
    : op_(op), mfem::Solver(op->Height(), true)
{
    buildLaplacians();
}

void DiracSmoother2D::buildLaplacians()
{
    const auto& d0   = op_->getExteriorDerivative(0);
    const auto& d1   = op_->getExteriorDerivative(1);
    const auto& del1 = op_->getCodifferential(1);
    const auto& del2 = op_->getCodifferential(2);

    L_[0].reset(mfem::Mult(del1, d0));

    {
        std::unique_ptr<mfem::SparseMatrix> L1_A(mfem::Mult(d0, del1));
        std::unique_ptr<mfem::SparseMatrix> L1_B(mfem::Mult(del2, d1));
        L_[1].reset(mfem::Add(*L1_A, *L1_B));
    }

    L_[2].reset(mfem::Mult(d1, del2));

    // =========================================================
    //         ENFORCE BCond::Essential BCs ON LAPLACIANS
    // =========================================================
    if (op_->getBCond() == BCond::Essential)
    {
        for (int k = 0; k < 3; ++k)
        {
            auto ess_dofs = op_->getEssentialTrueDofs(k);
            if (ess_dofs && ess_dofs->Size() > 0)
            {
                // Iterate through the boundary DOFs and eliminate the row/col
                // DIAG_ONE places 1.0 on the diagonal to keep it non-singular
                for (int i = 0; i < ess_dofs->Size(); ++i)
                {
                    L_[k]->EliminateRowCol(
                        (*ess_dofs)[i],
                        mfem::Operator::DiagonalPolicy::DIAG_ONE);
                }
            }
        }
    }

    // Initialize BlockDiagonalPreconditioners
    forw_ = std::make_unique<mfem::BlockDiagonalPreconditioner>(
        op_->getBlockOffsets());

    for (int k = 0; k < 3; ++k)
    {
        // Add GSSmoother configured for forward sweep (type = 1)
        forw_->SetDiagonalBlock(k, new mfem::GSSmoother(*L_[k], 1));
    }
}

void DiracSmoother2D::Mult(const mfem::Vector& rhs, mfem::Vector& x) const
{
    if (!this->iterative_mode)
        x = 0.0;

    mfem::BlockVector       bx(x.GetData(), op_->getBlockOffsets());
    const mfem::BlockVector brhs(
        const_cast<mfem::real_t*>(rhs.GetData()), op_->getBlockOffsets());

    mfem::BlockVector corr(op_->getBlockOffsets());

    // Calculate residual
    mfem::BlockVector res = op_->createBlockVector();
    op_->applyDecOp(bx, res);
    res -= rhs;

    // Apply forward Gauss-Seidel block preconditioner
    forw_->Mult(res, corr);

    // Distribute
    mfem::BlockVector& dist_corr = res;  // reference to save memory
    op_->applyDecOp(corr, dist_corr);
    bx -= dist_corr;

    // op_->eliminateTrivialSolutionHarmonics(bx);
}