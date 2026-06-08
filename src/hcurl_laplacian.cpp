#include "hcurl_laplacian.hpp"

using namespace HCurlLaplacian;

// =======================================================================
//                       VECTOR LAPLACIAN OPERATOR
// =======================================================================

void HCurlLaplacianOperator::applyDecOp(
    const mfem::BlockVector& in,
    mfem::BlockVector&       out) const
{
    out = 0.0;

    const auto& d0   = getExteriorDerivative(0);
    const auto& d1   = getExteriorDerivative(1);
    const auto& del1 = getCodifferential(1);
    const auto& del2 = getCodifferential(2);

    // out_0 = \delta_1 * in_1 - in_0
    del1.Mult(in.GetBlock(1), out.GetBlock(0));
    out.GetBlock(0) -= in.GetBlock(0);

    // out_1 = (\delta_2 * d_1) * in_1 + d_0 * in_0
    mfem::Vector d1_in1(d1.Height());
    d1.Mult(in.GetBlock(1), d1_in1);
    del2.Mult(d1_in1, out.GetBlock(1));
    d0.AddMult(in.GetBlock(0), out.GetBlock(1));
}

void HCurlLaplacianOperator::applyFemOp(
    const mfem::BlockVector& in,
    mfem::BlockVector&       out) const
{
    const auto& d0 = getExteriorDerivative(0);
    const auto& d1 = getExteriorDerivative(1);
    const auto& M0 = getMassMatrix(0);
    const auto& M1 = getMassMatrix(1);
    const auto& M2 = getMassMatrix(2);

    // out_0 = d_0^T * M_1 * in_1 - M_0 * in_0
    mfem::Vector M1_in1(M1->Height());
    M1->Mult(in.GetBlock(1), M1_in1);
    d0.MultTranspose(M1_in1, out.GetBlock(0));

    mfem::Vector M0_in0(M0->Height());
    M0->Mult(in.GetBlock(0), M0_in0);
    out.GetBlock(0) -= M0_in0;

    // out_1 = d_1^T * M_2 * d_1 * in_1 + M_1 * d_0 * in_0
    mfem::Vector d1_in1(d1.Height());
    d1.Mult(in.GetBlock(1), d1_in1);
    mfem::Vector M2_d1_in1(M2->Height());
    M2->Mult(d1_in1, M2_d1_in1);
    d1.MultTranspose(M2_d1_in1, out.GetBlock(1));

    mfem::Vector d0_in0(d0.Height());
    d0.Mult(in.GetBlock(0), d0_in0);
    mfem::Vector M1_d0_in0(M1->Height());
    M1->Mult(d0_in0, M1_d0_in0);
    out.GetBlock(1) += M1_d0_in0;
}

std::shared_ptr<mfem::SparseMatrix> HCurlLaplacianOperator::assembleDec()
{
    const mfem::SparseMatrix& d0   = this->getExteriorDerivative(0);
    const mfem::SparseMatrix& d1   = this->getExteriorDerivative(1);
    const mfem::SparseMatrix& del1 = this->getCodifferential(1);
    const mfem::SparseMatrix& del2 = this->getCodifferential(2);

    auto curlcurl = std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(del2, d1));

    mfem::SparseMatrix neg_I0(this->getBlockSize(0));
    for (int i = 0; i < neg_I0.Height(); ++i)
        neg_I0.Set(i, i, -1.0);

    // --- ENFORCE ESSENTIAL BCs (Diagonals Only) ---
    // d and delta rows/cols are already eliminated, so off-diagonals are safe.
    if (bc_ == BCond::Essential)
    {
        for (int i : *this->essential_dof_[0])
            neg_I0.Set(i, i, 1.0);
        for (int i : *this->essential_dof_[1])
            curlcurl->Set(i, i, 1.0);
    }
    neg_I0.Finalize();
    curlcurl->Finalize();

    mfem::BlockMatrix bmat(this->block_offsets_);
    bmat.SetBlock(0, 0, &neg_I0);
    bmat.SetBlock(0, 1, const_cast<mfem::SparseMatrix*>(&del1));
    bmat.SetBlock(1, 0, const_cast<mfem::SparseMatrix*>(&d0));
    bmat.SetBlock(1, 1, curlcurl.get());

    auto M = std::shared_ptr<mfem::SparseMatrix>(bmat.CreateMonolithic());
    return M;
}

std::shared_ptr<mfem::SparseMatrix> HCurlLaplacianOperator::assembleFem()
{
    mfem::BilinearForm m0(this->spaces_[0].get());
    m0.AddDomainIntegrator(new mfem::MassIntegrator());
    m0.Assemble();
    m0.Finalize();
    auto M0 = std::unique_ptr<mfem::SparseMatrix>(m0.LoseMat());

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

    // Negate M0 for the -in_0 term
    *M0 *= -1.0;

    // --- ENFORCE ESSENTIAL BCs (Block-by-Block Elimination) ---
    if (bc_ == BCond::Essential)
    {
        // 1. Eliminate dependent off-diagonal connections in Grad
        Grad->EliminateCols(*this->essential_dof_masks_[0]);  // 0-forms (Nodes)
        for (auto k : *this->essential_dof_[1])               // 1-forms (Edges)
            Grad->EliminateRow(k);

        // 2. Eliminate connections in the diagonal blocks and enforce 1.0 on
        // diagonal

        // Step A: Eliminate columns using the pre-built masks
        M0->EliminateCols(*this->essential_dof_masks_[0]);
        CurlCurl->EliminateCols(*this->essential_dof_masks_[1]);

        // Step B: Eliminate rows individually and inject 1.0 on the diagonal
        for (auto k : *this->essential_dof_[0])
            M0->EliminateRow(k, mfem::Operator::DiagonalPolicy::DIAG_ONE);

        for (auto k : *this->essential_dof_[1])
            CurlCurl->EliminateRow(k, mfem::Operator::DiagonalPolicy::DIAG_ONE);
    }

    auto Div = std::unique_ptr<mfem::SparseMatrix>(mfem::Transpose(*Grad));

    const int        num_blocks = (bc_ == BCond::Natural) ? 3 : 2;
    mfem::Array<int> aug_offsets(num_blocks + 1);
    aug_offsets[0] = 0;
    aug_offsets[1] = this->getBlockSize(0);
    aug_offsets[2] = aug_offsets[1] + this->getBlockSize(1);
    if (bc_ == BCond::Natural)
        aug_offsets[3] = aug_offsets[2] + 1;

    mfem::BlockMatrix bmat(this->block_offsets_);
    bmat.SetBlock(0, 0, M0.get());
    bmat.SetBlock(0, 1, Div.get());
    bmat.SetBlock(1, 0, Grad.get());
    bmat.SetBlock(1, 1, CurlCurl.get());

    const int          N_k = this->getBlockSize(0);
    mfem::SparseMatrix col_vec(N_k, 1);
    mfem::SparseMatrix row_vec(1, N_k);

    return std::shared_ptr<mfem::SparseMatrix>(bmat.CreateMonolithic());
}

// =======================================================================
//                       VECTOR LAPLACIAN SMOOTHER
// =======================================================================

// Simple utility to negate an operator for the block preconditioner
class NegatedOperator : public mfem::Operator
{
private:
    const mfem::Operator& op_;

public:
    NegatedOperator(const mfem::Operator& op)
        : mfem::Operator(op.Height(), op.Width()), op_(op)
    {
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override
    {
        op_.Mult(x, y);
        y.Neg();
    }

    void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override
    {
        op_.MultTranspose(x, y);
        y.Neg();
    }

    void AddMult(
        const mfem::Vector& x,
        mfem::Vector&       y,
        const mfem::real_t  a = 1.0) const override
    {
        op_.AddMult(x, y, -a);
    }

    void AddMultTranspose(
        const mfem::Vector& x,
        mfem::Vector&       y,
        const mfem::real_t  a = 1.0) const override
    {
        op_.AddMultTranspose(x, y, -a);
    }
};

HCurlLaplacianSmoother::HCurlLaplacianSmoother(
    std::shared_ptr<const HCurlLaplacianOperator> op)
    : mfem::Solver(op->Height()), op_(op)
{
    buildLaplacians();
}

void HCurlLaplacianSmoother::buildLaplacians()
{
    const mfem::SparseMatrix& d0   = op_->getExteriorDerivative(0);
    const mfem::SparseMatrix& d1   = op_->getExteriorDerivative(1);
    const mfem::SparseMatrix& del1 = op_->getCodifferential(1);
    const mfem::SparseMatrix& del2 = op_->getCodifferential(2);

    auto curlcurl_dec =
        std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(del2, d1));

    // L0pI = \delta_1 * d_0 + I
    L0pI_ = std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(del1, d0));
    mfem::Vector one(L0pI_->Height());
    one = 1.0;
    mfem::SparseMatrix I0(one);
    L0pI_ = std::unique_ptr<mfem::SparseMatrix>(mfem::Add(*L0pI_, I0));

    // L1 = d_0 * \delta_1 + curlcurl
    auto d0_del1 = std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(d0, del1));
    L1_ =
        std::unique_ptr<mfem::SparseMatrix>(mfem::Add(*d0_del1, *curlcurl_dec));

    if (op_->getBCond() == BCond::Essential)
    {
        auto ess_0 = op_->getEssentialTrueDofs(0);
        if (ess_0 && ess_0->Size() > 0)
            for (int i = 0; i < ess_0->Size(); ++i)
                L0pI_->EliminateRowCol((*ess_0)[i], mfem::Operator::DIAG_ONE);

        auto ess_1 = op_->getEssentialTrueDofs(1);
        if (ess_1 && ess_1->Size() > 0)
            for (int i = 0; i < ess_1->Size(); ++i)
                L1_->EliminateRowCol((*ess_1)[i], mfem::Operator::DIAG_ONE);
    }

    // 1. Instantiate the individual Gauss-Seidel block smoothers
    // For Mult (forward, lower triangular solve): use Forward GS (type = 1)
    gs0_forw_ = std::make_unique<mfem::GSSmoother>(*L0pI_, 1);
    gs1_forw_ = std::make_unique<mfem::GSSmoother>(*L1_, 1);

    gs0_forw_->iterative_mode = gs1_forw_->iterative_mode = false;

    // Define memory-managed negated operators for block matrix assignment
    neg_d0_   = std::make_unique<NegatedOperator>(d0);
    neg_del1_ = std::make_unique<NegatedOperator>(del1);

    // 2. Initialize Block Preconditioners
    // forw_ is Lower Triangular
    forw_ = std::make_unique<mfem::BlockLowerTriangularPreconditioner>(
        op_->getBlockOffsets());

    forw_->owns_blocks = 0;

    // Populate forw_ (Lower Triangular) - Off-diagonal is -d0
    forw_->SetDiagonalBlock(0, gs0_forw_.get());
    forw_->SetDiagonalBlock(1, gs1_forw_.get());
    forw_->SetBlock(1, 0, neg_d0_.get());

    // 3. Setup the Self-Adjoint Distribution Operator P
    // P = [ -I_0     delta_1 ]
    //     [  d_0     I_1     ]
    dist_    = std::make_unique<mfem::BlockOperator>(op_->getBlockOffsets());
    id0_     = std::make_unique<mfem::IdentityOperator>(op_->getBlockSize(0));
    id1_     = std::make_unique<mfem::IdentityOperator>(op_->getBlockSize(1));
    neg_id0_ = std::make_unique<NegatedOperator>(*id0_);

    dist_->SetBlock(0, 0, neg_id0_.get());
    dist_->SetBlock(0, 1, const_cast<mfem::SparseMatrix*>(&del1));
    dist_->SetBlock(1, 0, const_cast<mfem::SparseMatrix*>(&d0));
    dist_->SetBlock(1, 1, id1_.get());
}

void HCurlLaplacianSmoother::Mult(const mfem::Vector& rhs, mfem::Vector& x)
    const
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

    // Reads 'corr', writes to 'dist_corr' (aliased to res)
    dist_->Mult(corr, dist_corr);

    bx -= dist_corr;

    // op_->eliminateTrivialSolutionHarmonicsBlock(bx);
}
