#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <mfem.hpp>
#include <vector>

#include "SpectraErrorOp.hpp"

#define SPECTRA_TOL 1e-8

// Robust comparator for complex numbers to ensure stable sorting
// Primary key: Real part (Descending)
// Secondary key: Imaginary part (Descending)
bool ComplexCompare(
    const std::complex<mfem::real_t>& a,
    const std::complex<mfem::real_t>& b)
{
    if (std::abs(a.real() - b.real()) > 1e-12)
        return a.real() > b.real();
    return a.imag() > b.imag();
}

// Helper to filter top-k by magnitude, then sort by value for direct comparison
void PrepareEigenvalues(
    const Eigen::Vector<std::complex<mfem::real_t>, Eigen::Dynamic>& input,
    std::vector<std::complex<mfem::real_t>>&                         output,
    const int                                                        k)
{
    std::vector<std::complex<mfem::real_t>> temp;
    for (int i = 0; i < input.size(); ++i)
        temp.push_back(input(i));

    std::sort(
        temp.begin(), temp.end(),
        [](const std::complex<mfem::real_t>& a,
           const std::complex<mfem::real_t>& b)
        { return std::abs(a) > std::abs(b); });

    if (temp.size() > k)
        temp.resize(k);

    std::sort(temp.begin(), temp.end(), ComplexCompare);

    output = temp;
}

// Tests the solver against a diagonal system where eigenvalues are analytically
// known. Constructs a diagonal matrix A and sets P = I. The error operator is E
// = I - A, so eigenvalues should be exactly (1.0 - A_ii).
TEST(SpectraTest, DiagonalSystem)
{
    const int n       = 50;
    const int numEigs = 5;

    mfem::SparseMatrix                                        A(n, n);
    Eigen::Vector<std::complex<mfem::real_t>, Eigen::Dynamic> expected_vec(n);

    for (int i = 0; i < n; i++)
    {
        mfem::real_t val = 0.5 + 0.4 * ((mfem::real_t) i / (n - 1));
        A.Set(i, i, val);
        expected_vec(i) = std::complex<mfem::real_t>(1.0 - val, 0.0);
    }
    A.Finalize();

    mfem::IdentityOperator P(n);

    mfem::Vector v(n);
    v.Randomize(1);

    // Compute Spectra results passing the tolerance
    auto [spectra_res, ev] = errorOpEig(A, P, v, numEigs, SPECTRA_TOL);

    // Prepare both sets for comparison
    std::vector<std::complex<mfem::real_t>> expected_sorted;
    std::vector<std::complex<mfem::real_t>> actual_sorted;

    PrepareEigenvalues(expected_vec, expected_sorted, numEigs);
    PrepareEigenvalues(spectra_res, actual_sorted, numEigs);

    ASSERT_EQ(actual_sorted.size(), numEigs);

    for (int i = 0; i < numEigs; i++)
    {
        EXPECT_NEAR(
            actual_sorted[i].real(), expected_sorted[i].real(), SPECTRA_TOL);
        EXPECT_NEAR(
            actual_sorted[i].imag(), expected_sorted[i].imag(), SPECTRA_TOL);
    }
}

// Tests the solver against a dense random system.
// Uses Eigen::EigenSolver to compute the ground truth eigenvalues of (I - A).
// Verifies that Spectra's top-k eigenvalues match the Eigen results exactly.
TEST(SpectraTest, DenseRandomSystem)
{
    const int n       = 20;
    const int numEigs = 6;

    mfem::DenseMatrix                                           A(n);
    Eigen::Matrix<mfem::real_t, Eigen::Dynamic, Eigen::Dynamic> E_mat(n, n);

    srand(42);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mfem::real_t val = ((mfem::real_t) rand() / RAND_MAX) * 0.5;
            A(i, j)          = val;
            E_mat(i, j)      = (i == j ? 1.0 : 0.0) - val;
        }
    }

    // Ground Truth: Compute all eigenvalues using Eigen
    Eigen::EigenSolver<
        Eigen::Matrix<mfem::real_t, Eigen::Dynamic, Eigen::Dynamic>>
                                                              es(E_mat);
    Eigen::Vector<std::complex<mfem::real_t>, Eigen::Dynamic> all_evals =
        es.eigenvalues();

    mfem::IdentityOperator P(n);

    mfem::Vector v(n);
    v.Randomize(1);

    // Test Subject: Compute top k using Spectra, passing the tolerance
    auto [spectra_res, ev] = errorOpEig(A, P, v, numEigs, SPECTRA_TOL);

    // Prepare for comparison
    std::vector<std::complex<mfem::real_t>> expected_sorted;
    std::vector<std::complex<mfem::real_t>> actual_sorted;

    PrepareEigenvalues(all_evals, expected_sorted, numEigs);
    PrepareEigenvalues(spectra_res, actual_sorted, numEigs);

    ASSERT_EQ(actual_sorted.size(), numEigs);

    for (int i = 0; i < numEigs; i++)
    {
        EXPECT_NEAR(
            actual_sorted[i].real(), expected_sorted[i].real(), SPECTRA_TOL)
            << "Real part mismatch at index " << i;
        EXPECT_NEAR(
            actual_sorted[i].imag(), expected_sorted[i].imag(), SPECTRA_TOL)
            << "Imag part mismatch at index " << i;
    }
}
