#ifndef HCURL_LAPLACIAN_H
#define HCURL_LAPLACIAN_H

#include <memory>

#include "block_upper_triangular_preconditioner.hpp"
#include "mfem.hpp"
#include "operator.tpp"

namespace HCurlLaplacian
{

// =======================================================================
//                       VECTOR LAPLACIAN OPERATOR
// =======================================================================
class HCurlLaplacianOperator
    : public WhitneyFormOperator<HCurlLaplacianOperator, 0, 1>
{
public:
    HCurlLaplacianOperator(
        std::shared_ptr<Mesh> mesh,
        const BCond           BD   = BCond::Natural,
        const OperatorMode    mode = OperatorMode::DEC,
        const MassLumping     ml   = MassLumping::RowSum)
        : WhitneyFormOperator<HCurlLaplacianOperator, 0, 1>(mesh, BD, mode, ml)
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
    std::shared_ptr<mfem::SparseMatrix> assembleFem() override;
};

// =======================================================================
//                       VECTOR LAPLACIAN SMOOTHER
// =======================================================================
class HCurlLaplacianSmoother : public mfem::Solver
{
private:
    std::shared_ptr<const HCurlLaplacianOperator> op_;
    std::unique_ptr<mfem::SparseMatrix>           L0pI_;  // \delta_1 * d_0 + I
    std::unique_ptr<mfem::SparseMatrix> L1_;  // d_0 * \delta_1 + \delta_2 * d_1

    // Memory-managed smoothers
    std::unique_ptr<mfem::GSSmoother> gs0_forw_;
    std::unique_ptr<mfem::GSSmoother> gs1_forw_;
    std::unique_ptr<mfem::GSSmoother> gs0_back_;
    std::unique_ptr<mfem::GSSmoother> gs1_back_;

    // forw_ is Lower Triangular (used in Mult)
    std::unique_ptr<mfem::BlockLowerTriangularPreconditioner> forw_;

    // Self-adjoint distribution operator P and its internal components
    std::unique_ptr<mfem::BlockOperator>    dist_;
    std::unique_ptr<mfem::IdentityOperator> id0_;
    std::unique_ptr<mfem::IdentityOperator> id1_;
    std::unique_ptr<mfem::Operator>         neg_id0_;
    std::unique_ptr<mfem::Operator>         neg_d0_;
    std::unique_ptr<mfem::Operator>         neg_del1_;

    void buildLaplacians();

public:
    HCurlLaplacianSmoother(std::shared_ptr<const HCurlLaplacianOperator> op);

    void SetOperator(const mfem::Operator& op) override {}

    void Mult(const mfem::Vector& rhs, mfem::Vector& x) const override;
};

}  // namespace HCurlLaplacian

#endif  // VECLAP_H
