#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

#include "dirac_2d.hpp"
#include "mesh.hpp"
#include "mfem.hpp"
#include "mg.tpp"

using namespace Dirac2D;

// Instantiate the Multigrid template for the 2D Dirac system
// Active forms 0 (H1), 1 (HCurl), and 2 (L2)
typedef Multigrid<DiracOperator2D, DiracSmoother2D, 0, 1, 2> DiracMG;

// Helper to construct the RHS BlockVector using MFEM integrators for 3 spaces
mfem::BlockVector getLoadVector(
    std::shared_ptr<DiracOperator2D> op,
    mfem::FunctionCoefficient&       f0,
    mfem::VectorFunctionCoefficient& f1,
    mfem::FunctionCoefficient&       f2)
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

    // Block 2: L2 source
    mfem::LinearForm lf2(op->getSpace(2).get());
    lf2.AddDomainIntegrator(new mfem::DomainLFIntegrator(f2));
    lf2.Assemble();
    load.GetBlock(2) = lf2.GetData();

    // Apply boundary conditions (sets essential DOFs to zero in RHS)
    op->eliminateBC(load);
    // We need zero mean (just project, pray the error is not huge)
    op->applyInvHodgeStar(load);
    op->eliminateTrivialSolutionHarmonics(load);
    op->applyHodgeStar(load);

    return load;
}

int main(int argc, char* argv[])
{
    const int nref = 6;

#ifdef MFEM_USE_OPENMP
    mfem::Device device("omp");
#endif

    std::cout << std::scientific;

    // --- 1. Physics: Exact Solutions and Sources ---
    // Example lambdas (adjust these to match your specific Dirac test case)
    auto f0 = [](const mfem::Vector& v) -> mfem::real_t { return 0; };
    auto f1 = [](const mfem::Vector& v, mfem::Vector& w)
    {
        const mfem::real_t &x = v[0], y = v[1];
        w[0] = 0;
        w[1] = 4 * M_PI * std::sin(2 * M_PI * x) * std::cos(2 * M_PI * y);
    };
    auto f2 = [](const mfem::Vector& v) -> mfem::real_t
    {
        const mfem::real_t &x = v[0], y = v[1];
        return 2 * M_PI * (std::cos(2 * M_PI * x) - std::cos(2 * M_PI * y));
    };
    // The exact solution
    auto u0 = [](const mfem::Vector& v) -> mfem::real_t
    {
        const mfem::real_t &x = v(0), y = v(1);
        return std::sin(2 * M_PI * x) * std::sin(2 * M_PI * y);
    };
    auto u0_grad = [](const mfem::Vector& v, mfem::Vector& w)
    {
        const mfem::real_t &x = v(0), y = v(1);
        w[0] = 2 * M_PI * std::cos(2 * M_PI * x) * std::sin(2 * M_PI * y);
        w[1] = 2 * M_PI * std::sin(2 * M_PI * x) * std::cos(2 * M_PI * y);
    };
    auto u1 = [](const mfem::Vector& v, mfem::Vector& w)
    {
        const mfem::real_t &x = v[0], y = v[1];
        w[0] = std::sin(2 * M_PI * y);
        w[1] = std::sin(2 * M_PI * x);
    };
    auto u1_curl_vec = [&](const mfem::Vector& v, mfem::Vector& w)
    { w[0] = f2(v); };
    auto u2 = [](const mfem::Vector& v) -> mfem::real_t
    {
        const mfem::real_t &x = v[0], y = v[1];
        return std::cos(2 * M_PI * x) * std::cos(2 * M_PI * y);
    };

    // Placeholder source terms (f = Op * u_exact)
    mfem::FunctionCoefficient       u0_coeff(u0);
    mfem::VectorFunctionCoefficient u0_grad_coeff(2, u0_grad);
    mfem::VectorFunctionCoefficient u1_coeff(2, u1);
    mfem::VectorFunctionCoefficient u1_curl_coeff(1, u1_curl_vec);
    mfem::FunctionCoefficient       u2_coeff(u2);

    // Assuming we use the same functions for sources for the sake of the CVG
    // harness
    mfem::FunctionCoefficient       f0_coeff(f0);
    mfem::VectorFunctionCoefficient f1_coeff(2, f1);
    mfem::FunctionCoefficient       f2_coeff(f2);

    // --- 2. Setup Hierarchy ---
    mfem::Mesh m = mfem::Mesh("../../meshes/2d/square_centered.msh");
        //mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::TRIANGLE, true);
    auto mesh = std::make_shared<Mesh>(std::move(m));

    DiracMG MG(OperatorMode::Galerkin);
    MG.addCoarseLevel(
        nullptr, mesh, BCond::Essential, OperatorMode::Galerkin, MassLumping::RowSum);
    MG.setIterativeMode(false);

    // --- 3. Convergence Table Header ---
    std::cout << "\n" << std::string(105, '=') << "\n";
    std::cout << std::left << std::setw(6) << "Lvl" << std::right
              << std::setw(15) << "Unknowns" << std::right << std::setw(15)
              << "h" << std::right << std::setw(22) << "Sobolev Error"
              << std::right << std::setw(12) << "EOC" << std::right
              << std::setw(18) << "GMRES Iters" << "\n";
    std::cout << std::string(105, '-') << "\n";

    mfem::real_t errn_old = 0.0, h_old = 0.0;

    for (int lvl = 0; lvl <= nref; ++lvl)
    {
        auto               op = MG.getFinestOperator();
        const mfem::real_t h  = op->getMesh()->meshWidth();

        // Prepare System
        mfem::BlockVector rhs = getLoadVector(op, f0_coeff, f1_coeff, f2_coeff);
        mfem::BlockVector u   = op->createBlockVector();
        u                     = 0.0;

        // Solver
        mfem::FGMRESSolver gmres;
        gmres.SetAbsTol(1e-12);
        gmres.SetRelTol(1e-6);
        gmres.SetMaxIter(200);
        gmres.SetPrintLevel(0);

        gmres.SetPreconditioner(MG);
        gmres.SetOperator(*op);
        gmres.Mult(rhs, u);

        // Error Analysis
        mfem::GridFunction u0_gf(op->getSpace(0).get(), u.GetBlock(0));
        mfem::GridFunction u1_gf(op->getSpace(1).get(), u.GetBlock(1));
        mfem::GridFunction u2_gf(op->getSpace(2).get(), u.GetBlock(2));

        const mfem::real_t errn =
            u0_gf.ComputeH1Error(&u0_coeff, &u0_grad_coeff) +
            u1_gf.ComputeHCurlError(&u1_coeff, &u1_curl_coeff) +
            u2_gf.ComputeL2Error(u2_coeff);

        // Print Row
        std::cout << std::left << std::setw(6) << lvl << std::right
                  << std::setw(15) << op->Height() << std::right
                  << std::scientific << std::setprecision(4) << std::setw(15)
                  << h << std::right << std::scientific << std::setprecision(6)
                  << std::setw(22) << errn;

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

        if (lvl < nref)
            MG.addRefinedLevel();
    }
    std::cout << std::string(105, '=') << "\n" << std::endl;

    return 0;
}
