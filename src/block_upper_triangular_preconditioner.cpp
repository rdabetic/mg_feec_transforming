#include "block_upper_triangular_preconditioner.hpp"

namespace mfem
{

BlockUpperTriangularPreconditioner::BlockUpperTriangularPreconditioner(
    const Array<int>& offsets_)
    : Solver(offsets_.Last()),
      owns_blocks(0),
      nBlocks(offsets_.Size() - 1),
      offsets(0),
      ops(nBlocks, nBlocks)
{
    ops = nullptr;
    offsets.MakeRef(offsets_);
}

void BlockUpperTriangularPreconditioner::SetDiagonalBlock(
    int       iblock,
    Operator* op)
{
    MFEM_VERIFY(
        offsets[iblock + 1] - offsets[iblock] == op->Height() &&
            offsets[iblock + 1] - offsets[iblock] == op->Width(),
        "incompatible Operator dimensions");

    SetBlock(iblock, iblock, op);
}

void BlockUpperTriangularPreconditioner::SetBlock(
    int       iRow,
    int       iCol,
    Operator* op)
{
    MFEM_VERIFY(
        iRow <= iCol,
        "cannot set block in lower triangle for an upper triangular "
        "preconditioner");
    MFEM_VERIFY(
        offsets[iRow + 1] - offsets[iRow] == op->NumRows() &&
            offsets[iCol + 1] - offsets[iCol] == op->NumCols(),
        "incompatible Operator dimensions");

    ops(iRow, iCol) = op;
}

// Operator application (Backward Substitution)
void BlockUpperTriangularPreconditioner::Mult(const Vector& x, Vector& y) const
{
    MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
    MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

    x.Read();
    y.Write();
    y = 0.0;

    xblock.Update(const_cast<Vector&>(x), offsets);
    yblock.Update(y, offsets);

    // Backward substitution: start from bottom block and work up
    for (int iRow = nBlocks - 1; iRow >= 0; --iRow)
    {
        tmp.SetSize(offsets[iRow + 1] - offsets[iRow]);
        tmp.UseDevice(true);
        tmp2.SetSize(offsets[iRow + 1] - offsets[iRow]);
        tmp2.UseDevice(true);
        tmp2 = 0.0;
        tmp2 += xblock.GetBlock(iRow);

        // Multiply by upper blocks (columns to the right)
        for (int jCol = iRow + 1; jCol < nBlocks; ++jCol)
        {
            if (ops(iRow, jCol))
            {
                ops(iRow, jCol)->Mult(yblock.GetBlock(jCol), tmp);
                tmp2 -= tmp;
            }
        }
        if (ops(iRow, iRow))
            ops(iRow, iRow)->Mult(tmp2, yblock.GetBlock(iRow));
        else
            yblock.GetBlock(iRow) = tmp2;
    }

    for (int iRow = 0; iRow < nBlocks; ++iRow)
        yblock.GetBlock(iRow).SyncAliasMemory(y);
}

// Action of the transpose operator (Forward Substitution, since U^T is Lower
// Triangular)
void BlockUpperTriangularPreconditioner::MultTranspose(
    const Vector& x,
    Vector&       y) const
{
    MFEM_ASSERT(x.Size() == height, "incorrect input Vector size");
    MFEM_ASSERT(y.Size() == width, "incorrect output Vector size");

    x.Read();
    y.Write();
    y = 0.0;

    xblock.Update(const_cast<Vector&>(x), offsets);
    yblock.Update(y, offsets);

    // Forward substitution: start from top block and work down
    for (int iRow = 0; iRow < nBlocks; ++iRow)
    {
        tmp.SetSize(offsets[iRow + 1] - offsets[iRow]);
        tmp.UseDevice(true);
        tmp2.SetSize(offsets[iRow + 1] - offsets[iRow]);
        tmp2.UseDevice(true);
        tmp2 = 0.0;
        tmp2 += xblock.GetBlock(iRow);

        // Multiply by transpose of upper blocks (which are now lower blocks)
        for (int jCol = 0; jCol < iRow; ++jCol)
        {
            if (ops(jCol, iRow))  // Notice swapped indices for transpose access
            {
                ops(jCol, iRow)->MultTranspose(yblock.GetBlock(jCol), tmp);
                tmp2 -= tmp;
            }
        }
        if (ops(iRow, iRow))
            ops(iRow, iRow)->MultTranspose(tmp2, yblock.GetBlock(iRow));
        else
            yblock.GetBlock(iRow) = tmp2;
    }

    for (int iRow = 0; iRow < nBlocks; ++iRow)
        yblock.GetBlock(iRow).SyncAliasMemory(y);
}

BlockUpperTriangularPreconditioner::~BlockUpperTriangularPreconditioner()
{
    if (owns_blocks)
    {
        for (int iRow = 0; iRow < nBlocks; ++iRow)
            for (int jCol = 0; jCol < nBlocks; ++jCol)
                delete ops(iRow, jCol);
    }
}

}  // namespace mfem