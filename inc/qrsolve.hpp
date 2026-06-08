#ifndef EIGEN_SPARSE_QR_SOLVER_HPP
#define EIGEN_SPARSE_QR_SOLVER_HPP

#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include "mfem.hpp"

/**
 * @brief Direct Least Squares Solver using Eigen's SparseQR.
 * * Wraps Eigen::SparseQR in the mfem::Solver interface.
 */
class EigenSparseQRSolver : public mfem::Solver
{
private:
    // Eigen's SparseQR natively requires a column-major SparseMatrix
    Eigen::SparseMatrix<mfem::real_t> eigen_A_;
    Eigen::
        SparseQR<Eigen::SparseMatrix<mfem::real_t>, Eigen::COLAMDOrdering<int>>
            solver_;

public:
    EigenSparseQRSolver() : mfem::Solver() {}
    EigenSparseQRSolver(std::shared_ptr<mfem::SparseMatrix> mat)
        : mfem::Solver()
    {
        SetOperator(*mat);
    }

    /**
     * @brief Maps the MFEM Operator (must be a SparseMatrix) to Eigen and
     * computes the QR factorization.
     */
    void SetOperator(const mfem::Operator& op) override
    {
        // 1. Ensure the operator is actually an mfem::SparseMatrix
        const mfem::SparseMatrix* sparse_A =
            dynamic_cast<const mfem::SparseMatrix*>(&op);
        MFEM_VERIFY(
            sparse_A != nullptr,
            "EigenSparseQRSolver requires the operator to be an "
            "mfem::SparseMatrix.");

        height = op.Height();
        width  = op.Width();

        // 2. Safely read MFEM's raw arrays (handles GPU/CPU memory seamlessly
        // in modern MFEM)
        const int*          I    = sparse_A->GetI();
        const int*          J    = sparse_A->GetJ();
        const mfem::real_t* data = sparse_A->GetData();

        // 3. MFEM uses CSR (Row-Major) format. We construct an Eigen::Map to
        // wrap
        //    MFEM's memory without copying it.
        Eigen::Map<const Eigen::SparseMatrix<mfem::real_t, Eigen::RowMajor>>
            A_row(height, width, I[height], I, J, data);

        // 4. Convert CSR to CSC (Column-Major).
        //    Eigen's SparseQR strictly requires a column-major matrix, so a
        //    deep copy and transposition here is unavoidable.
        eigen_A_ = A_row;

        // 5. Compute the factorization
        solver_.compute(eigen_A_);
        MFEM_VERIFY(
            solver_.info() == Eigen::Success,
            "Eigen::SparseQR decomposition failed. Matrix may be invalid.");
    }

    /**
     * @brief Solves the least squares problem Ax ≈ b.
     */
    void Mult(const mfem::Vector& b, mfem::Vector& x) const override
    {
        MFEM_VERIFY(
            eigen_A_.rows() > 0, "Operator must be set before calling Mult().");
        MFEM_VERIFY(x.Size() == width, "Target vector has the wrong size!");

        // Map MFEM's vectors directly to Eigen (Zero-copy)
        Eigen::Map<const Eigen::Vector<mfem::real_t, Eigen::Dynamic>> eigen_b(
            b.GetData(), b.Size());
        Eigen::Map<Eigen::Vector<mfem::real_t, Eigen::Dynamic>> eigen_x(
            x.GetData(), x.Size());

        // Solve the system directly into x's memory
        eigen_x = solver_.solve(eigen_b);
        MFEM_VERIFY(
            solver_.info() == Eigen::Success, "Eigen::SparseQR solve failed.");
    }
};

#endif  // EIGEN_SPARSE_QR_SOLVER_HPP
