#ifndef MAGNETOSTATICS_H
#define MAGNETOSTATICS_H

#include <memory>

#include "block_upper_triangular_preconditioner.hpp"
#include "mfem.hpp"
#include "operator.tpp"

namespace Magnetostatics
{

// =======================================================================
//                    MAGNETOSTATIC OPERATOR
// =======================================================================
class MagOperator : public WhitneyFormOperator<MagOperator, 0, 1>
{
public:
    MagOperator(
        std::shared_ptr<Mesh> mesh,
        const BCond           BD   = BCond::Natural,
        const OperatorMode    mode = OperatorMode::DEC,
        const MassLumping     ml   = MassLumping::RowSum,
        const ScalarMassCoefficientArray& scalar_mass_coeffs = {},
        const MatrixMassCoefficientArray& matrix_mass_coeffs = {})
        : WhitneyFormOperator<MagOperator, 0, 1>(
              mesh, BD, mode, ml, scalar_mass_coeffs, matrix_mass_coeffs)
    {
        assembleSpace(2);
        assembleGalerkinMass(2);
        computeDecOperators(2);
        buildEssentialDofMask(2);
    }

    void applyDecOp(const mfem::BlockVector& in, mfem::BlockVector& out)
        const override;
    void applyFemOp(const mfem::BlockVector& in, mfem::BlockVector& out)
        const override;

    std::shared_ptr<mfem::SparseMatrix> assembleDec() override;
};

// =======================================================================
//                    MAGNETOSTATIC SMOOTHER
// =======================================================================
class MagSmoother : public mfem::Solver
{
private:
    std::shared_ptr<const MagOperator>  op_;
    std::unique_ptr<mfem::SparseMatrix> L0_;
    std::unique_ptr<mfem::SparseMatrix> L1_;

    // Memory-managed smoothers
    std::unique_ptr<mfem::GSSmoother> gs0_forw_;
    std::unique_ptr<mfem::GSSmoother> gs1_forw_;
    std::unique_ptr<mfem::GSSmoother> gs0_back_;
    std::unique_ptr<mfem::GSSmoother> gs1_back_;

    // forw_ is Upper Triangular (used in Mult)
    std::unique_ptr<mfem::BlockUpperTriangularPreconditioner> forw_;

    // Self-adjoint operator for the block distribution step
    std::unique_ptr<mfem::BlockOperator>    dist_;
    std::unique_ptr<mfem::IdentityOperator> id1_;

    void buildLaplacians();

public:
    MagSmoother(std::shared_ptr<const MagOperator> op);

    void SetOperator(const mfem::Operator& op) override {}

    void Mult(const mfem::Vector& rhs, mfem::Vector& x) const override;
};

}  // namespace Magnetostatics

#endif
