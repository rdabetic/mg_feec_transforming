// #include "SpectraErrorOp.hpp"

// #include <Spectra/GenEigsSolver.h>

// #include <algorithm>
// #include <complex>
// #include <iomanip>
// #include <iostream>

// ErrorOperator::ErrorOperator(
//     const mfem::Operator& mat,
//     const mfem::Operator& prec)
//     : mfem::Operator(mat.Height()), matOp(mat), precOp(prec),
//     zVec(mat.Height())
// {
//     MFEM_VERIFY(mat.Height() == mat.Width(), "Matrix must be square");
//     MFEM_VERIFY(prec.Height() == prec.Width(), "Preconditioner must be
//     square"); MFEM_VERIFY(
//         mat.Height() == prec.Height(),
//         "Matrix and Preconditioner dimensions must match");
// }

// void ErrorOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const
// {
//     y = x;
//     matOp.Mult(x, zVec);
//     precOp.AddMult(zVec, y, -1.0);
// }

// SpectraAdapter::SpectraAdapter(const mfem::Operator& op)
//     : mfemOp(op),
//       xVec(const_cast<mfem::real_t*>(static_cast<mfem::real_t*>(nullptr)),
//       0), yVec(static_cast<mfem::real_t*>(nullptr), 0)
// {
// }

// int SpectraAdapter::rows() const { return mfemOp.Height(); }

// int SpectraAdapter::cols() const { return mfemOp.Width(); }

// void SpectraAdapter::perform_op(const mfem::real_t* xIn, mfem::real_t* yOut)
// const
// {
//     xVec.SetDataAndSize(const_cast<mfem::real_t*>(xIn), mfemOp.Width());
//     yVec.SetDataAndSize(yOut, mfemOp.Height());

//     mfemOp.Mult(xVec, yVec);

//     xVec.SetDataAndSize(nullptr, 0);
//     yVec.SetDataAndSize(nullptr, 0);
// }

// /**
//  * Computes the largest eigenvalues of the Error Operator E = I - P*A.
//  * @param mat The system operator (A).
//  * @param prec The preconditioner/multigrid operator (P).
//  * @param init_guess A vector used as the starting point for the Krylov
//  * subspace. Must respect boundary conditions (e.g., 0 on essential
//  boundaries).
//  * @param numEigenvalues Number of eigenvalues to compute.
//  * @param tol Convergence tolerance for the Spectra solver.
//  * @param printResults Whether to print a formatted table to std::cout.
//  * @return Eigen::VectorXcd containing the converged complex eigenvalues.
//  */
// std::pair<Eigen::VectorXcd, Eigen::MatrixXcd>
// errorOpEig(
//     const mfem::Operator& mat,
//     const mfem::Operator& prec,
//     mfem::Vector&         init_guess,
//     const int             numEigenvalues,
//     const mfem::real_t          tol,
//     const bool            printResults)
// {
//     // 1. Setup the wrappers
//     ErrorOperator  errorOp(mat, prec);
//     SpectraAdapter spectraOp(errorOp);

//     // 2. Determine the number of Lanczos vectors (ncv)
//     const int ncv = std::min(
//         mat.Height(),
//         std::max(32, std::min(2 * numEigenvalues + 1, errorOp.Height())));

//     // 3. Initialize the Spectra solver
//     Spectra::GenEigsSolver<SpectraAdapter> eigs(spectraOp, numEigenvalues,
//     ncv);

//     // 4. Pass the initial guess (crucial for BCs)
//     eigs.init(init_guess.GetData());

//     // 5. Compute the eigenvalues
//     const int max_iters = 1000;
//     const int nConv =
//         eigs.compute(Spectra::SortRule::LargestMagn, max_iters, tol);

//     if (eigs.info() != Spectra::CompInfo::Successful)
//     {
//         std::cerr << "Spectra Error: Computation failed to converge.\n";
//         return {Eigen::VectorXcd(), Eigen::MatrixXcd()};
//     }

//     // Retrieve results
//     Eigen::VectorXcd vals = eigs.eigenvalues();
//     Eigen::MatrixXcd vecs = eigs.eigenvectors();

//     if (printResults)
//     {
//         // Save state to restore formatting later
//         std::ios oldState(nullptr);
//         oldState.copyfmt(std::cout);

//         std::cout << "\nSpectra: Computed " << nConv
//                   << " converged eigenpairs for the Error Operator.\n";
//         std::cout << std::string(85, '-') << "\n";

//         // Table Header
//         std::cout << std::left << std::setw(6) << "Idx"
//                   << std::right << std::setw(15) << "Real Part"
//                   << std::setw(20) << "Imag Part"
//                   << std::setw(18) << "Magnitude"
//                   << std::setw(22) << "Max E-vector Comp" << "\n";

//         std::cout << std::string(85, '-') << "\n";

//         std::cout << std::scientific << std::setprecision(6) << std::right;

//         for (int i = 0; i < vals.size(); i++)
//         {
//             std::complex<mfem::real_t> ev   = vals(i);
//             const char           sign = (ev.imag() >= 0) ? '+' : '-';

//             // Calculate max magnitude component of the corresponding
//             eigenvector mfem::real_t max_vec_comp =
//             vecs.col(i).array().abs().maxCoeff();

//             std::cout << std::left << std::setw(6) << i << std::right
//                       << std::setw(15) << ev.real() << "  " << sign << "  "
//                       << std::setw(13) << std::abs(ev.imag()) << "i"
//                       << std::setw(18) << std::abs(ev)
//                       << std::setw(22) << max_vec_comp << "\n";
//         }
//         std::cout << std::string(85, '-') << "\n";

//         std::cout.copyfmt(oldState);
//     }

//     return {vals, vecs};
// }
