#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "dirac_3d.hpp"
#include "mesh.hpp"
#include "mfem.hpp"
#include "mg.tpp"

using namespace Dirac3D;

/** * Typedef for the 3D Dirac Multigrid hierarchy.
 * Operates on the full de Rham sequence: 0-form (H1), 1-form (HCurl), 2-form
 * (HDiv), 3-form (L2).
 */
typedef Multigrid<DiracOperator3D, DiracSmoother3D, 0, 1, 2, 3> Dirac3DMG;

/**
 * Helper: Constructs the RHS BlockVector by assembling LinearForms for each
 * space.
 */
mfem::BlockVector getLoadVector(
    std::shared_ptr<DiracOperator3D> op,
    mfem::FunctionCoefficient&       f0,
    mfem::VectorFunctionCoefficient& f1,
    mfem::VectorFunctionCoefficient& f2,
    mfem::FunctionCoefficient&       f3)
{
    mfem::BlockVector load = op->createBlockVector();

    // Block 0: H1 source
    mfem::LinearForm lf0(op->getSpace(0).get());
    lf0.AddDomainIntegrator(new mfem::DomainLFIntegrator(f0));
    lf0.Assemble();
    load.GetBlock(0) = lf0.GetData();

    // Block 1: HCurl source
    mfem::LinearForm lf1(op->getSpace(1).get());
    lf1.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f1));
    lf1.Assemble();
    load.GetBlock(1) = lf1.GetData();

    // Block 2: HDiv source
    mfem::LinearForm lf2(op->getSpace(2).get());
    lf2.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f2));
    lf2.Assemble();
    load.GetBlock(2) = lf2.GetData();

    // Block 3: L2 source
    mfem::LinearForm lf3(op->getSpace(3).get());
    lf3.AddDomainIntegrator(new mfem::DomainLFIntegrator(f3));
    lf3.Assemble();
    load.GetBlock(3) = lf3.GetData();

    // Zero out entries corresponding to essential boundary conditions
    op->eliminateBC(load);
    // We need zero mean (just project, pray the error is not huge)
    op->applyInvHodgeStar(load);
    op->eliminateTrivialSolutionHarmonics(load);
    op->applyHodgeStar(load);

    return load;
}

// =======================================================================
//                                MAIN
// =======================================================================

int main(int argc, char* argv[])
{
    const int nref = 5;

#ifdef MFEM_USE_OPENMP
    mfem::Device device("omp");
#endif

    auto f0 = [](const mfem::Vector& v) -> mfem::real_t
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];
        return 6 * M_PI * std::sin(2 * M_PI * x) * std::sin(2 * M_PI * y) *
               std::sin(2 * M_PI * z);
    };
    auto f1 = [](const mfem::Vector& v, mfem::Vector& w)
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];
        w[0] = 2 * M_PI * std::sin(2 * M_PI * y) * std::sin(2 * M_PI * z) *
               std::cos(2 * M_PI * x);
        w[1] = 2 * M_PI * std::sin(2 * M_PI * x) * std::sin(2 * M_PI * z) *
               std::cos(2 * M_PI * y);
        w[2] = 2 * M_PI * std::sin(2 * M_PI * x) * std::sin(2 * M_PI * y) *
               std::cos(2 * M_PI * z);
    };
    auto f2 = [](const mfem::Vector& v, mfem::Vector& w)
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];
        w[0] = 2 * M_PI * std::sin(2 * M_PI * x) * std::cos(2 * M_PI * y) *
               std::cos(2 * M_PI * z);
        w[1] = 2 * M_PI * std::sin(2 * M_PI * y) * std::cos(2 * M_PI * x) *
               std::cos(2 * M_PI * z);
        w[2] = 2 * M_PI * std::sin(2 * M_PI * z) * std::cos(2 * M_PI * x) *
               std::cos(2 * M_PI * y);
    };
    auto f3 = [](const mfem::Vector& v) -> mfem::real_t
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];
        return 6 * M_PI * std::cos(2 * M_PI * x) * std::cos(2 * M_PI * y) *
               std::cos(2 * M_PI * z);
    };
    // The exact solution
    auto u0 = [](const mfem::Vector& v) -> mfem::real_t
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];
        return std::sin(2 * M_PI * x) * std::sin(2 * M_PI * y) *
               std::sin(2 * M_PI * z);
    };
    auto u0_grad = [](const mfem::Vector& v, mfem::Vector& w)
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];
        w[0] = 2 * M_PI * std::cos(2 * M_PI * x) * std::sin(2 * M_PI * y) *
               std::sin(2 * M_PI * z);
        w[1] = 2 * M_PI * std::sin(2 * M_PI * x) * std::cos(2 * M_PI * y) *
               std::sin(2 * M_PI * z);
        w[2] = 2 * M_PI * std::sin(2 * M_PI * x) * std::sin(2 * M_PI * y) *
               std::cos(2 * M_PI * z);
    };
    auto u1 = [](const mfem::Vector& v, mfem::Vector& w)
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];
        w[0] = std::cos(2 * M_PI * x) * std::sin(2 * M_PI * y) *
               std::sin(2 * M_PI * z);
        w[1] = std::sin(2 * M_PI * x) * std::cos(2 * M_PI * y) *
               std::sin(2 * M_PI * z);
        w[2] = std::sin(2 * M_PI * x) * std::sin(2 * M_PI * y) *
               std::cos(2 * M_PI * z);
    };
    auto u1_curl = [&](const mfem::Vector& v, mfem::Vector& w) { w = 0.; };
    auto u2      = [](const mfem::Vector& v, mfem::Vector& w)
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];
        w[0] = std::sin(2 * M_PI * x) * std::cos(2 * M_PI * y) *
               std::cos(2 * M_PI * z);
        w[1] = std::cos(2 * M_PI * x) * std::sin(2 * M_PI * y) *
               std::cos(2 * M_PI * z);
        w[2] = std::cos(2 * M_PI * x) * std::cos(2 * M_PI * y) *
               std::sin(2 * M_PI * z);
    };
    auto u2_div = [&](const mfem::Vector& v) -> mfem::real_t
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];
        return 6 * M_PI * cos(2 * x * M_PI) * cos(2 * y * M_PI) *
               cos(2 * z * M_PI);
    };
    auto u3 = [](const mfem::Vector& v) -> mfem::real_t
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];
        return std::cos(2 * M_PI * x) * std::cos(2 * M_PI * y) *
               std::cos(2 * M_PI * z);
    };

    // Setup exact solution coefficients
    mfem::FunctionCoefficient       u0_c(u0);
    mfem::VectorFunctionCoefficient u0_grad_c(3, u0_grad);
    mfem::FunctionCoefficient       f0_c(f0);

    mfem::VectorFunctionCoefficient u1_c(3, u1);
    mfem::VectorFunctionCoefficient u1_curl_c(3, u1_curl);
    mfem::VectorFunctionCoefficient f1_c(3, f1);

    mfem::VectorFunctionCoefficient u2_c(3, u2);
    mfem::FunctionCoefficient       u2_div_c(u2_div);
    mfem::VectorFunctionCoefficient f2_c(3, f2);

    mfem::FunctionCoefficient u3_c(u3);
    mfem::FunctionCoefficient f3_c(f3);

    // Initial coarse mesh
    mfem::Mesh m = mfem::Mesh("../../tests/cube.msh");
        //mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::TETRAHEDRON, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    // Initialize Multigrid Hierarchy
    Dirac3DMG MG(OperatorMode::Galerkin);
    MG.addCoarseLevel(
        nullptr, mesh, BCond::Essential, OperatorMode::Galerkin, MassLumping::RowSum);
    MG.setIterativeMode(false);

    // Output Header
    std::cout << "\n" << std::string(115, '=') << "\n";
    std::cout << std::left << std::setw(5) << "Lvl" << std::right
              << std::setw(15) << "DOFs" << std::right << std::setw(15) << "h"
              << std::right << std::setw(25) << "Dirac Sobolev Error"
              << std::right << std::setw(12) << "EOC" << std::right
              << std::setw(18) << "GMRES Iters" << "\n";
    std::cout << std::string(115, '-') << "\n";

    mfem::real_t errn_old = 0.0, h_old = 0.0;

    for (int lvl = 0; lvl <= nref; ++lvl)
    {
        auto               op = MG.getFinestOperator();
        const mfem::real_t h  = op->getMesh()->meshWidth();

        // 1. Setup system
        mfem::BlockVector rhs = getLoadVector(op, f0_c, f1_c, f2_c, f3_c);
        mfem::BlockVector u   = op->createBlockVector();
        u                     = 0.0;

        // 2. Solve with GMRES + Multigrid Preconditioner
        mfem::FGMRESSolver gmres;
        gmres.SetAbsTol(1e-12);
        gmres.SetRelTol(1e-6);
	gmres.SetKDim(32);
        gmres.SetMaxIter(128);

        gmres.SetPreconditioner(MG);
        gmres.SetOperator(*op);
        gmres.Mult(rhs, u);

        // 3. Sobolev Error Analysis (H1 + HCurl + HDiv + L2)
        mfem::GridFunction u0_gf(op->getSpace(0).get(), u.GetBlock(0));
        mfem::GridFunction u1_gf(op->getSpace(1).get(), u.GetBlock(1));
        mfem::GridFunction u2_gf(op->getSpace(2).get(), u.GetBlock(2));
        mfem::GridFunction u3_gf(op->getSpace(3).get(), u.GetBlock(3));

        const mfem::real_t errn = u0_gf.ComputeH1Error(&u0_c, &u0_grad_c) +
                                  u1_gf.ComputeHCurlError(&u1_c, &u1_curl_c) +
                                  u2_gf.ComputeHDivError(&u2_c, &u2_div_c) +
                                  u3_gf.ComputeL2Error(u3_c);

        // 4. Print results table row
        std::cout << std::left << std::setw(5) << lvl << std::right
                  << std::setw(15) << op->Height() << std::right
                  << std::scientific << std::setprecision(4) << std::setw(15)
                  << h << std::right << std::scientific << std::setprecision(6)
                  << std::setw(25) << errn;

        if (lvl > 0)
        {
            mfem::real_t eoc = std::log(errn_old / errn) / std::log(h_old / h);
            std::cout << std::right << std::fixed << std::setprecision(2)
                      << std::setw(12) << eoc;
        }
        else
        {
            std::cout << std::right << std::setw(12) << "---";
        }

        std::cout << std::right << std::setw(18) << gmres.GetNumIterations()
                  << std::endl;

        errn_old = errn;
        h_old    = h;

        // 5. Refine mesh and hierarchy
        if (lvl < nref)
            MG.addRefinedLevel();
    }
    std::cout << std::string(115, '=') << "\n" << std::endl;

    return 0;
}
