#ifndef DIRAC_2D_H
#define DIRAC_2D_H

#include <array>
#include <memory>

#include "mfem.hpp"
#include "operator.tpp"

namespace Dirac2D
{

enum SmootherType
{
    GAUSS_SEIDEL,
    JACOBI
};

// =======================================================================
//                       DIRAC OPERATOR (2D)
// =======================================================================
class DiracOperator2D : public WhitneyFormOperator<DiracOperator2D, 0, 1, 2>
{
public:
    DiracOperator2D(
        std::shared_ptr<Mesh> mesh,
        const BCond           BD   = BCond::Natural,
        const OperatorMode    mode = OperatorMode::DEC,
        const MassLumping     ml   = MassLumping::RowSum)
        : WhitneyFormOperator<DiracOperator2D, 0, 1, 2>(mesh, BD, mode, ml)
    {
    }

    void applyDecOp(const mfem::BlockVector& in, mfem::BlockVector& out)
        const override;
    void applyFemOp(const mfem::BlockVector& in, mfem::BlockVector& out)
        const override;

    std::shared_ptr<mfem::SparseMatrix> assembleDec() override;
    std::shared_ptr<mfem::SparseMatrix> assembleFem() override;
};

// =======================================================================
//                             SMOOTHER
// =======================================================================
class DiracSmoother2D : public mfem::Solver
{
private:
    std::shared_ptr<const DiracOperator2D>             op_;
    std::array<std::unique_ptr<mfem::SparseMatrix>, 3> L_;
    std::unique_ptr<mfem::BlockDiagonalPreconditioner> forw_;

public:
    DiracSmoother2D(std::shared_ptr<const DiracOperator2D> op);
    void SetOperator(const mfem::Operator& op) override {}

    void Mult(const mfem::Vector& rhs, mfem::Vector& x) const override;

private:
    void buildLaplacians();
};

}  // namespace Dirac2D

#endif
