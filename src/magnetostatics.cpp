#include "magnetostatics.hpp"

using namespace Magnetostatics;

// =======================================================================
//                    MAGNETOSTATIC OPERATOR
// =======================================================================

void MagOperator::applyDecOp(
    const mfem::BlockVector& in,
    mfem::BlockVector&       out) const
{
    out = 0.0;

    const auto& d0   = getExteriorDerivative(0);
    const auto& del1 = getCodifferential(1);
    const auto& del2 = getCodifferential(2);
    const auto& d1   = getExteriorDerivative(1);

    // out_0 = delta_1 * in_1 (Divergence-free constraint)
    del1.Mult(in.GetBlock(1), out.GetBlock(0));

    // out_1 = (delta_2 * d_1) * in_1 + d_0 * in_0 (Curl-Curl + Gradient)
    mfem::Vector d1_in1(d1.Height());
    d1.Mult(in.GetBlock(1), d1_in1);
    del2.Mult(d1_in1, out.GetBlock(1));
    d0.AddMult(in.GetBlock(0), out.GetBlock(1));

    // mfem::Vector bdin, bdout;
    // in.GetBlock(1).GetSubVector(
    //     *getEssentialTrueDofs(1), bdin
    //);
    // out.GetBlock(1).GetSubVector(
    //     *getEssentialTrueDofs(1), bdout
    //);
    // std::cout << bdin.Norml2() << " " << bdout.Norml2()
    //           << std::endl << std::endl;
    //
}

void MagOperator::applyFemOp(
    const mfem::BlockVector& in,
    mfem::BlockVector&       out) const
{
    out = 0.0;

    const auto& d0 = getExteriorDerivative(0);
    const auto& d1 = getExteriorDerivative(1);
    const auto& M1 = getMassMatrix(1);
    const auto& M2 = getMassMatrix(2);

    // out_0 = d_0^T * M_1 * in_1
    mfem::Vector M1_in1(M1->Height());
    M1->Mult(in.GetBlock(1), M1_in1);
    d0.MultTranspose(M1_in1, out.GetBlock(0));

    // out_1 = d_1^T * M_2 * d_1 * in_1 + M_1 * d_0 * in_0
    mfem::Vector d1_in1(d1.Height());
    d1.Mult(in.GetBlock(1), d1_in1);
    mfem::Vector M2_d1_in1(M2->Height());
    M2->Mult(d1_in1, M2_d1_in1);
    d1.MultTranspose(M2_d1_in1, out.GetBlock(1));

    mfem::Vector d0_in0(d0.Height());
    d0.Mult(in.GetBlock(0), d0_in0);
    M1->AddMult(d0_in0, out.GetBlock(1));
}

std::shared_ptr<mfem::SparseMatrix> MagOperator::assembleDec()
{
    const mfem::SparseMatrix& d0   = this->getExteriorDerivative(0);
    const mfem::SparseMatrix& d1   = this->getExteriorDerivative(1);
    const mfem::SparseMatrix& del1 = this->getCodifferential(1);
    const mfem::SparseMatrix& del2 = this->getCodifferential(2);

    auto curlcurl = std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(del2, d1));

    if (bc_ == BCond::Essential)
        for (int i : *this->essential_dof_[1])
            curlcurl->Set(i, i, 1.0);
    curlcurl->Finalize();

    mfem::BlockMatrix bmat(this->block_offsets_);
    bmat.SetBlock(0, 1, const_cast<mfem::SparseMatrix*>(&del1));
    bmat.SetBlock(1, 0, const_cast<mfem::SparseMatrix*>(&d0));
    bmat.SetBlock(1, 1, curlcurl.get());

    return std::shared_ptr<mfem::SparseMatrix>(bmat.CreateMonolithic());
}

std::shared_ptr<mfem::SparseMatrix> MagOperator::assembleFem()
{
    mfem::MixedBilinearForm grad(
        this->spaces_[0].get(), this->spaces_[1].get());
    grad.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
    grad.Assemble();
    grad.Finalize();
    auto Grad = std::unique_ptr<mfem::SparseMatrix>(grad.LoseMat());

    mfem::BilinearForm curlcurl(this->spaces_[1].get());
    curlcurl.AddDomainIntegrator(new mfem::CurlCurlIntegrator());
    curlcurl.Assemble();
    curlcurl.Finalize();
    auto CurlCurl = std::unique_ptr<mfem::SparseMatrix>(curlcurl.LoseMat());

    mfem::SparseMatrix zero0(this->getBlockSize(0));

    if (bc_ == BCond::Essential)
    {
        Grad->EliminateCols(*this->essential_dof_masks_[0]);
        for (auto k : *this->essential_dof_[1])
            Grad->EliminateRow(k);
        for (auto k : *this->essential_dof_[0])
            zero0.Set(k, k, 1.0);
        CurlCurl->EliminateCols(*this->essential_dof_masks_[1]);
        for (auto k : *this->essential_dof_[1])
            CurlCurl->EliminateRow(k, mfem::Operator::DiagonalPolicy::DIAG_ONE);
    }
    zero0.Finalize();

    auto Div = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*Grad));

    mfem::BlockMatrix bmat(this->block_offsets_);
    bmat.SetBlock(0, 0, &zero0);
    bmat.SetBlock(0, 1, Div.get());
    bmat.SetBlock(1, 0, Grad.get());
    bmat.SetBlock(1, 1, CurlCurl.get());

    return std::shared_ptr<mfem::SparseMatrix>(bmat.CreateMonolithic());
}

// =======================================================================
//                    MAGNETOSTATIC SMOOTHER
// =======================================================================

MagSmoother::MagSmoother(std::shared_ptr<const MagOperator> op)
    : mfem::Solver(op->Height(), true), op_(op)
{
    buildLaplacians();
}

void MagSmoother::buildLaplacians()
{
    const mfem::SparseMatrix& d0   = op_->getExteriorDerivative(0);
    const mfem::SparseMatrix& d1   = op_->getExteriorDerivative(1);
    const mfem::SparseMatrix& del1 = op_->getCodifferential(1);
    const mfem::SparseMatrix& del2 = op_->getCodifferential(2);

    auto curlcurl_dec =
        std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(del2, d1));

    // L0 = delta_1 * d_0 (Scalar Laplacian)
    L0_ = std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(del1, d0));

    // L1 = d_0 * delta_1 + delta_2 * d_1 (Vector Laplacian)
    auto d0_del1 = std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(d0, del1));
    L1_ =
        std::unique_ptr<mfem::SparseMatrix>(mfem::Add(*d0_del1, *curlcurl_dec));

    if (op_->getBCond() == BCond::Essential)
    {
        auto ess_0 = op_->getEssentialTrueDofs(0);
        if (ess_0)
            for (int i = 0; i < ess_0->Size(); ++i)
                L0_->EliminateRowCol((*ess_0)[i], mfem::Operator::DIAG_ONE);

        auto ess_1 = op_->getEssentialTrueDofs(1);
        if (ess_1)
            for (int i = 0; i < ess_1->Size(); ++i)
                L1_->EliminateRowCol((*ess_1)[i], mfem::Operator::DIAG_ONE);
    }

    // 1. Instantiate the individual Gauss-Seidel block smoothers
    // For Mult (forward algorithm direction): we use Backward GS (type = 2)
    gs0_forw_ = std::make_unique<mfem::GSSmoother>(*L0_, 2);
    gs1_forw_ = std::make_unique<mfem::GSSmoother>(*L1_, 2);

    gs0_forw_->iterative_mode = gs1_forw_->iterative_mode = false;

    // 2. Initialize Block Preconditioners
    // forw_ uses the Upper Triangular system
    forw_ = std::make_unique<mfem::BlockUpperTriangularPreconditioner>(
        op_->getBlockOffsets());

    forw_->owns_blocks = 0;

    // Populate forw_ (Upper Triangular)
    forw_->SetDiagonalBlock(0, gs0_forw_.get());
    forw_->SetDiagonalBlock(1, gs1_forw_.get());
    forw_->SetBlock(0, 1, const_cast<mfem::SparseMatrix*>(&del1));

    // 3. Initialize the Self-Adjoint Block Distribution Operator P
    dist_ = std::make_unique<mfem::BlockOperator>(op_->getBlockOffsets());
    id1_  = std::make_unique<mfem::IdentityOperator>(op_->getBlockSize(1));

    dist_->SetBlock(0, 1, const_cast<mfem::SparseMatrix*>(&del1));
    dist_->SetBlock(1, 0, const_cast<mfem::SparseMatrix*>(&d0));
    dist_->SetBlock(1, 1, id1_.get());
}

void MagSmoother::Mult(const mfem::Vector& rhs, mfem::Vector& x) const
{
    if (!this->iterative_mode)
        x = 0.0;

    mfem::BlockVector bx(x.GetData(), op_->getBlockOffsets());
    mfem::BlockVector res = op_->createBlockVector();

    op_->applyDecOp(bx, res);
    res -= rhs;

    mfem::BlockVector corr(op_->getBlockOffsets());

    forw_->Mult(res, corr);

    mfem::BlockVector& dist_corr = res;
    dist_->Mult(corr, dist_corr);

    bx -= dist_corr;

    // op_->eliminateTrivialSolutionHarmonicsBlock(bx);
}
