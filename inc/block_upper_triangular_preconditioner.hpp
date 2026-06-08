#ifndef BLOCK_UPPER_TRIANGULAR_PRECONDITIONER_HPP
#define BLOCK_UPPER_TRIANGULAR_PRECONDITIONER_HPP

#include "mfem.hpp"

namespace mfem
{

/**
 * @brief A block upper triangular preconditioner.
 * * This class mirrors mfem::BlockLowerTriangularPreconditioner but performs
 * backward substitution for the Mult() operation (upper triangular solve)
 * and forward substitution for the MultTranspose() operation.
 */
class BlockUpperTriangularPreconditioner : public mfem::Solver
{
private:
    int                            nBlocks;
    mfem::Array<int>               offsets;
    mfem::Array2D<mfem::Operator*> ops;

    mutable mfem::BlockVector xblock;
    mutable mfem::BlockVector yblock;
    mutable mfem::Vector      tmp;
    mutable mfem::Vector      tmp2;

public:
    int owns_blocks;

    /// Construct with a given set of block offsets
    BlockUpperTriangularPreconditioner(const mfem::Array<int>& offsets_);

    /// Set the diagonal block (must be square and match block dimensions)
    void SetDiagonalBlock(int iblock, mfem::Operator* op);

    /// Set an off-diagonal block (iRow <= iCol)
    void SetBlock(int iRow, int iCol, mfem::Operator* op);

    /// Required by mfem::Solver interface
    void SetOperator(const mfem::Operator& op) override {}

    /// Apply the preconditioner (Backward substitution)
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /// Apply the transpose preconditioner (Forward substitution)
    void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

    /// Controls block ownership (if non-zero, the preconditioner deletes them
    /// in the destructor)
    void OwnsBlocks(int owns) { owns_blocks = owns; }

    ~BlockUpperTriangularPreconditioner() override;
};

}  // namespace mfem

#endif  // BLOCK_UPPER_TRIANGULAR_PRECONDITIONER_HPP