#include <iostream>
#include <memory>

#include "magnetostatics.hpp"
#include "mesh.hpp"
#include "mfem.hpp"
#include "mg.tpp"

using namespace Magnetostatics;
typedef Multigrid<MagOperator, MagSmoother, 0, 1> MagMG;

// Helper to construct the RHS BlockVector using MFEM integrators
mfem::BlockVector getLoadVector(
    std::shared_ptr<MagOperator>     op,
    mfem::FunctionCoefficient&       f0,
    mfem::VectorFunctionCoefficient& f1)
{
    mfem::BlockVector load = op->createBlockVector();

    // Block 0: Scalar potential source
    mfem::LinearForm lf0(op->getSpace(0).get());
    lf0.AddDomainIntegrator(new mfem::DomainLFIntegrator(f0));
    lf0.Assemble();
    load.GetBlock(0) = lf0.GetData();

    // Block 1: Current density source
    mfem::LinearForm lf1(op->getSpace(1).get());
    lf1.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f1));
    lf1.Assemble();
    load.GetBlock(1) = lf1.GetData();

    // Set essential DOFs to 0 in RHS
    op->eliminateBC(load);
    // We need zero mean (just project, pray the error is not huge)
    op->applyInvHodgeStar(load);
    op->eliminateTrivialSolutionHarmonics(load);
    op->applyHodgeStar(load);

    return load;
}

int main(int argc, char* argv[])
{
    const int nref = 4;

#ifdef MFEM_USE_OPENMP
    mfem::Device device("omp");
#endif
    // device.Print(std::cout);
    std::cout << std::scientific;

    // The exact solution
    auto u0      = [](const mfem::Vector& v) -> mfem::real_t { return 0; };
    auto u0_grad = [](const mfem::Vector& v, mfem::Vector& w) { w = 0.; };
    auto u1      = [](const mfem::Vector& v, mfem::Vector& w)
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];
        w[0] = std::cos(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z);
        w[1] = -std::sin(M_PI * x) * std::cos(M_PI * y) * std::sin(M_PI * z);
        w[2] = 0;
    };
    auto u1_curl = [&](const mfem::Vector& v, mfem::Vector& w)
    {
        const mfem::real_t &x = v[0], y = v[1], z = v[2];

        w[0] =
            M_PI * std::sin(M_PI * x) * std::cos(M_PI * y) * std::cos(M_PI * z);
        w[1] =
            M_PI * std::cos(M_PI * x) * std::sin(M_PI * y) * std::cos(M_PI * z);
        w[2] = -2. * M_PI * std::cos(M_PI * x) * std::cos(M_PI * y) *
               std::sin(M_PI * z);
    };
    // Load `functions`
    auto f0 = [&](const mfem::Vector& v) -> mfem::real_t { return 0; };

    auto f1 = [&](const mfem::Vector& v, mfem::Vector& w)
    {
        u1(v, w);
        w *= 3. * std::pow(M_PI, 2);
    };
    // Make coefficients out of the lambdas of the exact solutions
    mfem::FunctionCoefficient       u0_coeff(u0);
    mfem::VectorFunctionCoefficient u0_grad_coeff(3, u0_grad);
    mfem::VectorFunctionCoefficient u1_coeff(3, u1);
    mfem::VectorFunctionCoefficient u1_curl_coeff(3, u1_curl);

    mfem::FunctionCoefficient       f0_coeff(f0);
    mfem::VectorFunctionCoefficient f1_coeff(3, f1);

    // 2. Setup Mesh
    mfem::Mesh mesh_mfem("../../tests/cube.msh");
    auto       mesh = std::make_shared<Mesh>(std::move(mesh_mfem));

    // Initialize Multigrid with the operator-only logic
    // We pass 'true' for use_fem, but we will call op->Mult() instead of
    // assembleFem()
    MagMG MG(OperatorMode::Galerkin);
    MG.addCoarseLevel(
        nullptr, mesh, BCond::Essential, OperatorMode::Galerkin, MassLumping::RowSum);
    MG.setIterativeMode(false);

    // --- NEW HEADER ---
    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << std::left << std::setw(6) << "Lvl" << std::right
              << std::setw(15) << "Unknowns" << std::right << std::setw(15)
              << "h" << std::right << std::setw(20) << "Sobolev Error"
              << std::right << std::setw(12) << "EOC" << std::right
              << std::setw(18) << "GMRES Iters" << "\n";
    std::cout << std::string(90, '-') << "\n";

    mfem::real_t errn_old = 0.0, h_old = 0.0;

    for (int lvl = 0; lvl <= nref; ++lvl)
    {
        auto               op           = MG.getFinestOperator();
        auto               current_mesh = op->getMesh();
        const mfem::real_t h            = current_mesh->meshWidth();

        mfem::BlockVector rhs = getLoadVector(op, f0_coeff, f1_coeff);
        mfem::BlockVector u   = op->createBlockVector();
        u                     = 0.0;

        mfem::FGMRESSolver gmres;
        gmres.SetAbsTol(1e-12);
        gmres.SetRelTol(1e-6);
        gmres.SetMaxIter(300);
        gmres.SetKDim(32);

        // Setting print level to 0 keeps the table clean.
        // Set to 1 if you want to see the residual per iteration.
        gmres.SetPrintLevel(0);

        gmres.SetPreconditioner(MG);
        gmres.SetOperator(*op);

        gmres.Mult(rhs, u);

        mfem::GridFunction u0_gf(op->getSpace(0).get(), u.GetBlock(0));
        mfem::GridFunction u1_gf(op->getSpace(1).get(), u.GetBlock(1));

        const mfem::real_t errn =
            u0_gf.ComputeH1Error(&u0_coeff, &u0_grad_coeff) +
            u1_gf.ComputeHCurlError(&u1_coeff, &u1_curl_coeff);

        // --- NEW ROW PRINTING ---
        std::cout << std::left << std::setw(6) << lvl << std::right
                  << std::setw(15) << op->Height() << std::right
                  << std::scientific << std::setprecision(4) << std::setw(15)
                  << h << std::right << std::scientific << std::setprecision(6)
                  << std::setw(20) << errn;

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
    std::cout << std::string(90, '=') << "\n" << std::endl;

    return 0;
}
