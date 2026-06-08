#ifndef SPECTRA_ERROR_OP_HPP
#define SPECTRA_ERROR_OP_HPP

#include <Spectra/GenEigsSolver.h>

#include <Eigen/Core>
#include <algorithm>
#include <complex>
#include <iomanip>
#include <iostream>
#include <mfem.hpp>
#include <utility>

/**
 * Templated Error Operator E = I - P*A
 * Supports physics-specific harmonic/kernel elimination.
 */
template <typename OpType, typename PrecType>
class ErrorOperator : public mfem::Operator
{
private:
    const OpType&                      matOp;
    const PrecType&                    precOp;
    mutable mfem::Vector               zVec;
    std::function<void(mfem::Vector&)> post_;

public:
    ErrorOperator(
        const OpType&                      mat,
        const PrecType&                    prec,
        std::function<void(mfem::Vector&)> post_proc = [](mfem::Vector&) {})
        : mfem::Operator(mat.Height()),
          matOp(mat),
          precOp(prec),
          zVec(mat.Height()),
          post_(post_proc)
    {
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override
    {
        // E = I - P*A
        y = x;
        matOp.Mult(x, zVec);
        precOp.AddMult(zVec, y, -1.0);
        // Post-processing (dealing with rounding errors)
        post_(y);
    }
};

/**
 * Adapter for Spectra to interface with the templated ErrorOperator.
 */
template <typename ErrorOpType>
class SpectraAdapter
{
private:
    const ErrorOpType&   mfemOp;
    mutable mfem::Vector xVec;
    mutable mfem::Vector yVec;

public:
    using Scalar = mfem::real_t;

    SpectraAdapter(const ErrorOpType& op)
        : mfemOp(op),
          xVec(
              const_cast<mfem::real_t*>(static_cast<mfem::real_t*>(nullptr)),
              0),
          yVec(static_cast<mfem::real_t*>(nullptr), 0)
    {
    }

    int rows() const { return mfemOp.Height(); }
    int cols() const { return mfemOp.Width(); }

    void perform_op(const mfem::real_t* xIn, mfem::real_t* yOut) const
    {
        xVec.SetDataAndSize(const_cast<mfem::real_t*>(xIn), mfemOp.Width());
        yVec.SetDataAndSize(yOut, mfemOp.Height());

        mfemOp.Mult(xVec, yVec);

        xVec.SetDataAndSize(nullptr, 0);
        yVec.SetDataAndSize(nullptr, 0);
    }
};

/**
 * Templated Eigenvalue Solver
 */
template <typename OpType, typename PrecType>
std::pair<
    Eigen::Vector<std::complex<mfem::real_t>, Eigen::Dynamic>,
    Eigen::Matrix<std::complex<mfem::real_t>, Eigen::Dynamic, Eigen::Dynamic>>
errorOpEig(
    const OpType&                      mat,
    const PrecType&                    prec,
    mfem::Vector&                      init_guess,
    const int                          numEigenvalues = 1,
    const mfem::real_t                 tol            = 1e-3,
    std::function<void(mfem::Vector&)> post           = [](mfem::Vector&) {},
    const bool                         printResults   = false)
{
    // 1. Setup wrappers
    ErrorOperator<OpType, PrecType>                 errorOp(mat, prec, post);
    SpectraAdapter<ErrorOperator<OpType, PrecType>> spectraOp(errorOp);

    // // 2. ncv rule: ncv > 2*k
    // const int ncv = std::min(
    //     mat.Height(),
    //     std::max(32, std::min(2 * numEigenvalues + 1, mat.Height())));

    // 2. ncv rule: ncv > 2*k
    const int ncv = std::min(mat.Height(), 32 * numEigenvalues + 8);

    // 3. Solver initialization
    Spectra::GenEigsSolver<SpectraAdapter<ErrorOperator<OpType, PrecType>>>
        eigs(spectraOp, numEigenvalues, ncv);

    // Ensure initial guess is clean before entering Krylov space
    // mat.eliminateTrivialSolutionHarmonics(init_guess);
    eigs.init(init_guess.GetData());

    // 4. Computation
    const int max_iters = 10'000;
    const int nConv =
        eigs.compute(Spectra::SortRule::LargestMagn, max_iters, tol);

    if (eigs.info() != Spectra::CompInfo::Successful)
    {
        throw std::runtime_error(
            "Spectra Error: Computation failed to converge.\n");
        return {
            Eigen::Vector<std::complex<mfem::real_t>, Eigen::Dynamic>(),
            Eigen::Matrix<
                std::complex<mfem::real_t>, Eigen::Dynamic, Eigen::Dynamic>()};
    }

    Eigen::Vector<std::complex<mfem::real_t>, Eigen::Dynamic> vals =
        eigs.eigenvalues();
    Eigen::Matrix<std::complex<mfem::real_t>, Eigen::Dynamic, Eigen::Dynamic>
        vecs = eigs.eigenvectors();

    if (printResults)
    {
        std::ios oldState(nullptr);
        oldState.copyfmt(std::cout);

        std::cout << "\nSpectra: Computed " << nConv
                  << " converged eigenpairs.\n";
        std::cout << std::string(85, '-') << "\n";
        std::cout << std::left << std::setw(6) << "Idx" << std::right
                  << std::setw(15) << "Real Part" << std::setw(20)
                  << "Imag Part" << std::setw(18) << "Magnitude"
                  << std::setw(22) << "Max E-vector Comp"
                  << "\n";
        std::cout << std::string(85, '-') << "\n";

        std::cout << std::scientific << std::setprecision(6) << std::right;

        for (int i = 0; i < vals.size(); i++)
        {
            std::complex<mfem::real_t> ev   = vals(i);
            const char                 sign = (ev.imag() >= 0) ? '+' : '-';
            mfem::real_t max_vec_comp = vecs.col(i).array().abs().maxCoeff();

            std::cout << std::left << std::setw(6) << i << std::right
                      << std::setw(15) << ev.real() << "  " << sign << "  "
                      << std::setw(13) << std::abs(ev.imag()) << "i"
                      << std::setw(18) << std::abs(ev) << std::setw(22)
                      << max_vec_comp << "\n";
        }
        std::cout << std::string(85, '-') << "\n";
        std::cout.copyfmt(oldState);
    }

    return {vals, vecs};
}

#endif  // SPECTRA_ERROR_OP_HPP
