#include "dirac_3d.hpp"

using namespace Dirac3D;

// =======================================================================
//                       DIRAC OPERATOR 3D
// =======================================================================

void DiracOperator3D::applyDecOp(
    const mfem::BlockVector& in,
    mfem::BlockVector&       out) const
{
    // D = d + \delta
    // out_0 = \delta_1 in_1
    // out_1 = d_0 in_0 + \delta_2 in_2
    // out_2 = d_1 in_1 + \delta_3 in_3
    // out_3 = d_2 in_2

    const auto& d0 = getExteriorDerivative(0);
    const auto& d1 = getExteriorDerivative(1);
    const auto& d2 = getExteriorDerivative(2);

    const auto& del1 = getCodifferential(1);
    const auto& del2 = getCodifferential(2);
    const auto& del3 = getCodifferential(3);

    del1.Mult(in.GetBlock(1), out.GetBlock(0));

    d0.Mult(in.GetBlock(0), out.GetBlock(1));
    del2.AddMult(in.GetBlock(2), out.GetBlock(1));

    d1.Mult(in.GetBlock(1), out.GetBlock(2));
    del3.AddMult(in.GetBlock(3), out.GetBlock(2));

    d2.Mult(in.GetBlock(2), out.GetBlock(3));
}

void DiracOperator3D::applyFemOp(
    const mfem::BlockVector& in,
    mfem::BlockVector&       out) const
{
    out = 0.0;

    // Direct reference binding (no casting needed)
    const mfem::SparseMatrix& d0 = this->getExteriorDerivative(0);
    const mfem::SparseMatrix& d1 = this->getExteriorDerivative(1);
    const mfem::SparseMatrix& d2 = this->getExteriorDerivative(2);

    const auto M1 = this->getMassMatrix(1);
    const auto M2 = this->getMassMatrix(2);
    const auto M3 = this->getMassMatrix(3);

    mfem::Vector tmp1(M1->Height());
    mfem::Vector tmp2(M2->Height());
    mfem::Vector tmp3(M3->Height());

    // 1. out_1 += M_1 * d_0 * in_0
    mfem::Vector d0_in0(d0.Height());
    d0.Mult(in.GetBlock(0), d0_in0);
    M1->SpMat().Mult(d0_in0, tmp1);
    out.GetBlock(1) += tmp1;

    // 2. out_0 += d_0^T * M_1 * in_1
    M1->SpMat().Mult(in.GetBlock(1), tmp1);
    d0.MultTranspose(tmp1, out.GetBlock(0));

    // 3. out_2 += M_2 * d_1 * in_1
    mfem::Vector d1_in1(d1.Height());
    d1.Mult(in.GetBlock(1), d1_in1);
    M2->SpMat().Mult(d1_in1, tmp2);
    out.GetBlock(2) += tmp2;

    // 4. out_1 += d_1^T * M_2 * in_2
    M2->SpMat().Mult(in.GetBlock(2), tmp2);
    d1.AddMultTranspose(tmp2, out.GetBlock(1));

    // 5. out_3 += M_3 * d_2 * in_2
    mfem::Vector d2_in2(d2.Height());
    d2.Mult(in.GetBlock(2), d2_in2);
    M3->SpMat().Mult(d2_in2, tmp3);
    out.GetBlock(3) += tmp3;

    // 6. out_2 += d_2^T * M_3 * in_3
    M3->SpMat().Mult(in.GetBlock(3), tmp3);
    d2.AddMultTranspose(tmp3, out.GetBlock(2));
}

std::shared_ptr<mfem::SparseMatrix> DiracOperator3D::assembleDec()
{
    // Fetch the pre-eliminated discrete operators
    const mfem::SparseMatrix& d0 = this->getExteriorDerivative(0);
    const mfem::SparseMatrix& d1 = this->getExteriorDerivative(1);
    const mfem::SparseMatrix& d2 = this->getExteriorDerivative(2);

    const mfem::SparseMatrix& del1 = this->getCodifferential(1);
    const mfem::SparseMatrix& del2 = this->getCodifferential(2);
    const mfem::SparseMatrix& del3 = this->getCodifferential(3);

    auto block_mat = std::make_unique<mfem::BlockMatrix>(this->block_offsets_);

    // Set off-diagonal blocks
    block_mat->SetBlock(0, 1, const_cast<mfem::SparseMatrix*>(&del1));
    block_mat->SetBlock(1, 0, const_cast<mfem::SparseMatrix*>(&d0));
    block_mat->SetBlock(1, 2, const_cast<mfem::SparseMatrix*>(&del2));
    block_mat->SetBlock(2, 1, const_cast<mfem::SparseMatrix*>(&d1));
    block_mat->SetBlock(2, 3, const_cast<mfem::SparseMatrix*>(&del3));
    block_mat->SetBlock(3, 2, const_cast<mfem::SparseMatrix*>(&d2));

    // --- ENFORCE ESSENTIAL BCs (Diagonals only) ---
    std::unique_ptr<mfem::SparseMatrix> diag0, diag1, diag2, diag3;
    if (bc_ == BCond::Essential)
    {
        diag0 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(0));
        for (int i : *this->essential_dof_[0])
            diag0->Set(i, i, 1.0);
        diag0->Finalize();
        block_mat->SetBlock(0, 0, diag0.get());

        diag1 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(1));
        for (int i : *this->essential_dof_[1])
            diag1->Set(i, i, 1.0);
        diag1->Finalize();
        block_mat->SetBlock(1, 1, diag1.get());

        diag2 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(2));
        for (int i : *this->essential_dof_[2])
            diag2->Set(i, i, 1.0);
        diag2->Finalize();
        block_mat->SetBlock(2, 2, diag2.get());

        diag3 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(3));
        for (int i : *this->essential_dof_[3])
            diag3->Set(i, i, 1.0);
        diag3->Finalize();
        block_mat->SetBlock(3, 3, diag3.get());
    }

    auto M = std::shared_ptr<mfem::SparseMatrix>(block_mat->CreateMonolithic());
    M->Finalize();

    return M;
}

std::shared_ptr<mfem::SparseMatrix> DiracOperator3D::assembleFem()
{
    // Assemble Mixed Bilinear Forms representing M*d
    auto bD0 = std::make_unique<mfem::MixedBilinearForm>(
        this->spaces_[0].get(), this->spaces_[1].get());
    auto bD1 = std::make_unique<mfem::MixedBilinearForm>(
        this->spaces_[1].get(), this->spaces_[2].get());
    auto bD2 = std::make_unique<mfem::MixedBilinearForm>(
        this->spaces_[2].get(), this->spaces_[3].get());

    mfem::ConstantCoefficient one(1.0);
    bD0->AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator(one));
    bD1->AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator(one));
    bD2->AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator(one));

    bD0->Assemble();
    bD0->Finalize();
    bD1->Assemble();
    bD1->Finalize();
    bD2->Assemble();
    bD2->Finalize();

    auto D0 = std::unique_ptr<mfem::SparseMatrix>(bD0->LoseMat());
    auto D1 = std::unique_ptr<mfem::SparseMatrix>(bD1->LoseMat());
    auto D2 = std::unique_ptr<mfem::SparseMatrix>(bD2->LoseMat());

    // --- ENFORCE ESSENTIAL BCs (Off-diagonal Elimination) ---
    if (bc_ == BCond::Essential)
    {
        // For Vertex DOFs (0-forms): Zero Row i of D0^T by zeroing Col i of D0
        D0->EliminateCols(*this->essential_dof_masks_[0]);

        // For Edge DOFs (1-forms):
        for (auto k : *this->essential_dof_[1])
            D0->EliminateRow(k);  // Zero Row j of D0
        D1->EliminateCols(
            *this->essential_dof_masks_[1]);  // Zero Row j of D1^T

        // For Face DOFs (2-forms):
        for (auto k : *this->essential_dof_[2])
            D1->EliminateRow(k);  // Zero Row m of D1
        D2->EliminateCols(
            *this->essential_dof_masks_[2]);  // Zero Row m of D2^T

        // For Cell DOFs (3-forms):
        for (auto k : *this->essential_dof_[3])
            D2->EliminateRow(k);  // Zero Row n of D2
    }

    auto D0T = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*D0));
    auto D1T = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*D1));
    auto D2T = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*D2));

    auto block_mat = std::make_unique<mfem::BlockMatrix>(this->block_offsets_);

    // Set off-diagonal blocks
    block_mat->SetBlock(0, 1, D0T.get());
    block_mat->SetBlock(1, 0, D0.get());
    block_mat->SetBlock(1, 2, D1T.get());
    block_mat->SetBlock(2, 1, D1.get());
    block_mat->SetBlock(2, 3, D2T.get());
    block_mat->SetBlock(3, 2, D2.get());

    // --- ENFORCE ESSENTIAL BCs (Diagonals) ---
    std::unique_ptr<mfem::SparseMatrix> diag0, diag1, diag2, diag3;
    if (bc_ == BCond::Essential)
    {
        diag0 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(0));
        for (int i : *this->essential_dof_[0])
            diag0->Set(i, i, 1.0);
        diag0->Finalize();
        block_mat->SetBlock(0, 0, diag0.get());

        diag1 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(1));
        for (int i : *this->essential_dof_[1])
            diag1->Set(i, i, 1.0);
        diag1->Finalize();
        block_mat->SetBlock(1, 1, diag1.get());

        diag2 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(2));
        for (int i : *this->essential_dof_[2])
            diag2->Set(i, i, 1.0);
        diag2->Finalize();
        block_mat->SetBlock(2, 2, diag2.get());

        diag3 = std::make_unique<mfem::SparseMatrix>(this->getBlockSize(3));
        for (int i : *this->essential_dof_[3])
            diag3->Set(i, i, 1.0);
        diag3->Finalize();
        block_mat->SetBlock(3, 3, diag3.get());
    }

    auto M = std::shared_ptr<mfem::SparseMatrix>(block_mat->CreateMonolithic());
    M->Finalize();

    return M;
}

// =======================================================================
//                       DIRAC SMOOTHER 3D
// =======================================================================

DiracSmoother3D::DiracSmoother3D(std::shared_ptr<const DiracOperator3D> op)
    : mfem::Solver(op->Height(), true), op_(op)
{
    buildLaplacians();
}

void DiracSmoother3D::buildLaplacians()
{
    const auto& d0 = op_->getExteriorDerivative(0);
    const auto& d1 = op_->getExteriorDerivative(1);
    const auto& d2 = op_->getExteriorDerivative(2);

    const auto& del1 = op_->getCodifferential(1);
    const auto& del2 = op_->getCodifferential(2);
    const auto& del3 = op_->getCodifferential(3);

    // L_0 = \delta_1 d_0
    L_[0].reset(mfem::Mult(del1, d0));

    // L_1 = d_0 \delta_1 + \delta_2 d_1
    {
        std::unique_ptr<mfem::SparseMatrix> L1_A(mfem::Mult(d0, del1));
        std::unique_ptr<mfem::SparseMatrix> L1_B(mfem::Mult(del2, d1));
        L_[1].reset(mfem::Add(*L1_A, *L1_B));
    }

    // L_2 = d_1 \delta_2 + \delta_3 d_2
    {
        std::unique_ptr<mfem::SparseMatrix> L2_A(mfem::Mult(d1, del2));
        std::unique_ptr<mfem::SparseMatrix> L2_B(mfem::Mult(del3, d2));
        L_[2].reset(mfem::Add(*L2_A, *L2_B));
    }

    // L_3 = d_2 \delta_3
    L_[3].reset(mfem::Mult(d2, del3));

    // Enforce Essential BCs on Laplacians
    if (op_->getBCond() == BCond::Essential)
    {
        for (int k = 0; k < 4; ++k)
        {
            auto ess_dofs = op_->getEssentialTrueDofs(k);
            if (ess_dofs && ess_dofs->Size() > 0)
            {
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

    for (int k = 0; k < 4; ++k)
    {
        // Add GSSmoother configured for forward sweep (type = 1)
        forw_->SetDiagonalBlock(k, new mfem::GSSmoother(*L_[k], 1));
    }
}

void DiracSmoother3D::Mult(const mfem::Vector& rhs, mfem::Vector& x) const
{
    if (!this->iterative_mode)
        x = 0.0;

    mfem::BlockVector bx(x.GetData(), op_->getBlockOffsets());
    mfem::BlockVector brhs(
        const_cast<mfem::real_t*>(rhs.GetData()), op_->getBlockOffsets());

    mfem::BlockVector corr(op_->getBlockOffsets());

    mfem::BlockVector res(op_->getBlockOffsets());

    // Calculate residual
    op_->applyDecOp(bx, res);
    res -= rhs;

    // Apply forward Gauss-Seidel block preconditioner
    forw_->Mult(res, corr);

    // Distribute
    mfem::BlockVector& distributed_corr =
        res;  // save memory, res not needed anymore
    op_->applyDecOp(corr, distributed_corr);
    bx -= distributed_corr;

    // op_->eliminateTrivialSolutionHarmonicsBlock(bx);
}
