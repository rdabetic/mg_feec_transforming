#include "transfer.hpp"

TransferOperator::TransferOperator(
    const mfem::FiniteElementSpace&         coarse_fe,
    const mfem::FiniteElementSpace&         fine_fe,
    const mfem::Vector&                     star_c,
    const mfem::Vector&                     star_f,
    std::shared_ptr<const mfem::Array<int>> ess_c,
    std::shared_ptr<const mfem::Array<int>> ess_f)
    : mfem::Operator(fine_fe.GetTrueVSize(), coarse_fe.GetTrueVSize()),
      ess_c_(std::move(ess_c)),
      ess_f_(std::move(ess_f)),
      star_c_(star_c),
      star_f_(star_f),
      tmp_(fine_fe.GetTrueVSize())
{
    // mfem::OperatorPtr P_ptr(mfem::Operator::MFEM_SPARSEMAT);
    mfem::OperatorPtr P_ptr(mfem::Operator::ANY_TYPE);
    fine_fe.GetTransferOperator(coarse_fe, P_ptr);
    P_ptr.SetOperatorOwner(false);

    // If shared pointers are provided, apply the essential BC constraints
    if (ess_c_ && ess_f_)
    {
        inner_P_.reset(P_ptr.Ptr());
        P_ = std::make_unique<mfem::RectangularConstrainedOperator>(
            inner_P_.get(), *ess_c_, *ess_f_);
    }
    else
    {
        P_.reset(P_ptr.Ptr());
    }
}

void TransferOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    P_->Mult(x, y);
}

void TransferOperator::MultTranspose(const mfem::Vector& x, mfem::Vector& y)
    const
{
    // Multiply by fine Hodge star (M_f)
    tmp_ = x;
    tmp_ *= star_f_;

    // Apply standard matrix transpose (P^T)
    P_->MultTranspose(tmp_, y);

    // Multiply by inverse coarse Hodge star (M_c^{-1})
    y /= star_c_;
}
