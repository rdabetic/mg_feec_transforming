#ifndef TRANSFER_OPERATOR_H
#define TRANSFER_OPERATOR_H

#include <memory>

#include "mfem.hpp"

class TransferOperator : public mfem::Operator
{
private:
    std::shared_ptr<const mfem::Array<int>> ess_c_;
    std::shared_ptr<const mfem::Array<int>> ess_f_;
    std::unique_ptr<mfem::Operator>         inner_P_;
    std::unique_ptr<mfem::Operator>         P_;
    const mfem::Vector&                     star_c_;
    const mfem::Vector&                     star_f_;
    mutable mfem::Vector                    tmp_;

public:
    TransferOperator(
        const mfem::FiniteElementSpace&         coarse_fe,
        const mfem::FiniteElementSpace&         fine_fe,
        const mfem::Vector&                     star_c,
        const mfem::Vector&                     star_f,
        std::shared_ptr<const mfem::Array<int>> ess_c = nullptr,
        std::shared_ptr<const mfem::Array<int>> ess_f = nullptr);

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;
};

#endif
