#ifndef DIRAC_3D_H
#define DIRAC_3D_H

#include <array>
#include <memory>

#include "mfem.hpp"
#include "operator.tpp"

namespace Dirac3D
{

enum SmootherType
{
    GAUSS_SEIDEL,
    JACOBI
};

// =======================================================================
//                       DIRAC OPERATOR 3D
// =======================================================================
class DiracOperator3D : public WhitneyFormOperator<DiracOperator3D, 0, 1, 2, 3>
{
public:
    DiracOperator3D(
        std::shared_ptr<Mesh> mesh,
        const BCond           BD   = BCond::Natural,
        const OperatorMode    mode = OperatorMode::DEC,
        const MassLumping     ml   = MassLumping::RowSum)
        : WhitneyFormOperator<DiracOperator3D, 0, 1, 2, 3>(mesh, BD, mode, ml)
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
//                       DIRAC SMOOTHER 3D
// =======================================================================
class DiracSmoother3D : public mfem::Solver
{
private:
    std::shared_ptr<const DiracOperator3D>             op_;
    std::array<std::unique_ptr<mfem::SparseMatrix>, 4> L_;
    std::unique_ptr<mfem::BlockDiagonalPreconditioner> forw_;

    void buildLaplacians();

public:
    DiracSmoother3D(std::shared_ptr<const DiracOperator3D> op);

    void SetOperator(const mfem::Operator& op) override {}

    void Mult(const mfem::Vector& rhs, mfem::Vector& x) const override;
};

}  // namespace Dirac3D

#endif  // DIRAC_3D_H
