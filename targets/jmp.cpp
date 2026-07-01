#include <boost/program_options.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "SpectraErrorOp.hpp"
#include "hcurl_laplacian.hpp"
#include "mesh.hpp"
#include "mfem.hpp"
#include "mg.tpp"

namespace po = boost::program_options;
using namespace HCurlLaplacian;

namespace
{
struct BenchResult
{
    int          refinements;
    mfem::real_t mesh_width;
    int          dofs;
    std::string  spectral_radii;
    mfem::real_t avg_gmres_iters;
    mfem::real_t mg_asymptotic_rate;
};

struct CliOptions
{
    std::string mesh_file = "../../tests/cube.msh";
    std::string output_file = "hcurl_jump_cvg.csv";
    std::string bcond = "essential";
    std::string cycle = "v";
    std::string lumping = "rowsum";
    int         max_ref = 3;
    int         num_runs = 8;
    int         krylov_dim = 32;
    int         gmres_maxit = 256;
    int         mg_iters = 16;
    int         pre_smooth = 1;
    int         post_smooth = 1;
    int         verbose = 1;
    double      gmres_tol = 1e-6;
    double      jump_value = 1e3;
    double      inner_cube_size = 0.35;
};

void save_to_csv(const std::string& filename, const BenchResult& res)
{
    std::ifstream check(filename);
    const bool exists = check.peek() != std::ifstream::traits_type::eof();
    check.close();

    std::ofstream file(filename, std::ios::app);
    if (!exists)
        file << "Refinements,mesh-width,DOFs,SpectralRadii,AvgGMRESIters,MGAsymptoticRate\n";

    file << res.refinements << "," << std::scientific << std::setprecision(6)
         << res.mesh_width << "," << res.dofs << ",\"" << res.spectral_radii
         << "\"," << std::fixed << std::setprecision(2)
         << res.avg_gmres_iters << "," << std::scientific
         << std::setprecision(6) << res.mg_asymptotic_rate << "\n";
}

CliOptions parse_options(int argc, char* argv[])
{
    CliOptions opts;

    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Show help message")(
        "mesh,m", po::value<std::string>(&opts.mesh_file)->default_value(opts.mesh_file),
        "Mesh file (default: unit cube mesh)")(
        "output,o", po::value<std::string>(&opts.output_file)->default_value(opts.output_file),
        "CSV output file")(
        "bcond", po::value<std::string>(&opts.bcond)->default_value(opts.bcond),
        "essential, natural")(
        "cycle,c", po::value<std::string>(&opts.cycle)->default_value(opts.cycle),
        "v, w")(
        "lumping,l", po::value<std::string>(&opts.lumping)->default_value(opts.lumping),
        "rowsum, scaledid")(
        "max-ref,r", po::value<int>(&opts.max_ref)->default_value(opts.max_ref),
        "Max refinements")(
        "num-runs,n", po::value<int>(&opts.num_runs)->default_value(opts.num_runs),
        "GMRES runs per level")(
        "krylov,k", po::value<int>(&opts.krylov_dim)->default_value(opts.krylov_dim),
        "GMRES Krylov dimension")(
        "maxit,N", po::value<int>(&opts.gmres_maxit)->default_value(opts.gmres_maxit),
        "GMRES max iterations")(
        "tol,t", po::value<double>(&opts.gmres_tol)->default_value(opts.gmres_tol),
        "GMRES relative tolerance")(
        "mg-iters,M", po::value<int>(&opts.mg_iters)->default_value(opts.mg_iters),
        "Number of multigrid residual-decay iterations")(
        "pre-smooth", po::value<int>(&opts.pre_smooth)->default_value(opts.pre_smooth),
        "Pre-smoothing iterations")(
        "post-smooth", po::value<int>(&opts.post_smooth)->default_value(opts.post_smooth),
        "Post-smoothing iterations")(
        "jump", po::value<double>(&opts.jump_value)->default_value(opts.jump_value),
        "Coefficient jump value inside the inner cube")(
        "inner-size", po::value<double>(&opts.inner_cube_size)->default_value(opts.inner_cube_size),
        "Side length of the inner cube jump region")(
        "verbose,V", po::value<int>(&opts.verbose)->default_value(opts.verbose),
        "Verbose residual decay output");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        std::exit(0);
    }

    return opts;
}

template <typename MGType>
BenchResult run_level(
    MGType&                MG,
    const int              level,
    const int              num_runs,
    const mfem::real_t     gmres_tol,
    const int              krylov_dim,
    const int              gmres_maxit,
    const int              mg_iters,
    const int              verbose,
    const BCond            bc,
    const OperatorMode     mode,
    const int              num_eigs = 1,
    const mfem::real_t     eig_tol = 1e-2)
{
    auto op = MG.getFinestOperator();

    // Compute spectral radii (DEC mode)
    std::stringstream ss_csv;
    MG.setMode(OperatorMode::DEC);
    op->setMode(OperatorMode::DEC);
    auto start_v = op->getRandomVector(42 + level);
    auto post = [&](mfem::Vector& x) {
        mfem::BlockVector bx(x.GetData(), op->getBlockOffsets());
        op->eliminateTrivialSolutionHarmonics(bx);
        if (bc == BCond::Essential)
            op->eliminateBC(bx);
    };

    auto [ew, ev] = errorOpEig(*op, MG, start_v, num_eigs, eig_tol, post);
    for (int i = 0; i < ew.size(); ++i) {
        mfem::real_t mag = std::abs(ew[i]);
        ss_csv << mag << (i == ew.size() - 1 ? "" : " ");
    }

    MG.setMode(OperatorMode::Galerkin);
    op->setMode(OperatorMode::Galerkin);

    int total_iters = 0;
    for (int r = 0; r < num_runs; ++r)
    {
        mfem::BlockVector x_true = op->getRandomVector(12345 + 17 * level + r);
        op->eliminateTrivialSolutionHarmonics(x_true);
        if (bc == BCond::Essential)
            op->eliminateBC(x_true);

        mfem::BlockVector rhs(op->getBlockOffsets());
        mfem::BlockVector x   = op->createBlockVector();
        x = 0.0;
        op->applyFemOp(x_true, rhs);

        mfem::FGMRESSolver gmres;
        gmres.SetOperator(*op);
        gmres.SetPreconditioner(MG);
        gmres.SetAbsTol(0.0);
        gmres.SetRelTol(gmres_tol);
        gmres.SetMaxIter(gmres_maxit);
        gmres.SetKDim(krylov_dim);
        gmres.SetPrintLevel(-1);
        gmres.Mult(rhs, x);
        total_iters += gmres.GetNumIterations();
    }

    const mfem::real_t avg_iters =
        static_cast<mfem::real_t>(total_iters) / std::max(1, num_runs);

    MG.setMode(OperatorMode::DEC);
    MG.setIterativeMode(true);
    op->setMode(OperatorMode::DEC);

    mfem::real_t mg_rate = 1.0;
    if (mg_iters > 0)
    {
        mfem::BlockVector rhs(op->getBlockOffsets());
        mfem::BlockVector x = op->getRandomVector(54321 + level);
        mfem::BlockVector rhs_rand = op->getRandomVector(12345 + level);
        op->eliminateTrivialSolutionHarmonics(rhs_rand);
        op->applyDecOp(rhs_rand, rhs);

        mfem::BlockVector res = op->createBlockVector();
        const mfem::real_t rhs_norm = std::max<mfem::real_t>(rhs.Norml2(), 1e-16);

        std::vector<mfem::real_t> norms;
        norms.reserve(mg_iters);
        mfem::real_t prev_norm = -1.0;

        if (verbose)
        {
            std::cout << "------------------------------------------------------\n";
            std::cout << "  MG RESIDUAL DECAY\n";
            std::cout << "------------------------------------------------------\n";
            std::cout << " " << std::left << std::setw(8) << "Iter"
                      << std::setw(24) << "Rel. Residual"
                      << "Reduction Factor\n";
            std::cout << "---------------------------------------------"
                         "---------\n";
        }

        for (int it = 0; it < mg_iters; ++it)
        {
            MG.Mult(rhs, x);
            op->Mult(x, res);
            res -= rhs;

            const mfem::real_t nrm = res.Norml2() / rhs_norm;
            norms.push_back(nrm);
            const mfem::real_t red = (prev_norm > 0.0) ? nrm / prev_norm : 0.0;

            if (verbose)
            {
                std::cout << " " << std::left << std::setw(8) << it
                          << std::scientific << std::setprecision(6)
                          << std::setw(24) << nrm;
                if (prev_norm > 0.0)
                {
                    std::cout << std::fixed << std::setprecision(4) << red;
                }
                else
                {
                    std::cout << "-----";
                }
                std::cout << "\n";
            }

            prev_norm = nrm;
        }

        if (verbose)
            std::cout << "---------------------------------------------"
                         "---------\n";

        const int tail = std::min(4, static_cast<int>(norms.size()) / 2);
        const int start = std::max(0, static_cast<int>(norms.size()) - tail);
        mfem::real_t log_sum = 0.0;
        int count = 0;
        for (int i = start; i + 1 < static_cast<int>(norms.size()); ++i)
        {
            if (norms[i] > 1e-15 && norms[i + 1] > 1e-15)
            {
                log_sum += std::log(norms[i + 1] / norms[i]);
                ++count;
            }
        }
        if (count > 0)
            mg_rate = std::exp(log_sum / count);

        if (verbose)
        {
            std::cout << "  Iterations:      " << mg_iters << "\n";
            std::cout << "  Final Rel. Res.: " << std::scientific
                      << std::setprecision(6)
                      << (norms.empty() ? 0.0 : norms.back()) << "\n";
            std::cout << "  Asymptotic Rate: " << std::fixed
                      << std::setprecision(6) << mg_rate << "\n";
        }

        MG.setIterativeMode(false);
        MG.setMode(OperatorMode::Galerkin);
        op->setMode(OperatorMode::Galerkin);
    }

    return {
        level, op->getMesh()->meshWidth(), op->Height(), ss_csv.str(),
        avg_iters, mg_rate
    };
}

}  // namespace

int main(int argc, char* argv[])
{
#ifdef MFEM_USE_OPENMP
    mfem::Device device("omp");
    device.Print(std::cout);
#endif

    const CliOptions opts = parse_options(argc, argv);

    if (opts.bcond != "essential" && opts.bcond != "natural")
        throw std::invalid_argument(
            "Invalid bcond: expected 'essential' or 'natural'.");
    if (opts.cycle != "v" && opts.cycle != "w")
        throw std::invalid_argument("Invalid cycle: expected 'v' or 'w'.");
    if (opts.lumping != "rowsum" && opts.lumping != "scaledid")
        throw std::invalid_argument(
            "Invalid lumping: expected 'rowsum' or 'scaledid'.");

    const BCond bc = (opts.bcond == "essential") ? BCond::Essential
                                                    : BCond::Natural;
    const CycleType cycle_type = (opts.cycle == "v") ? CycleType::VCYCLE
                                                      : CycleType::WCYCLE;
    const MassLumping lumping = (opts.lumping == "scaledid")
                                    ? MassLumping::ScaledId
                                    : MassLumping::RowSum;

    std::cout << std::scientific;
    std::cout << "=================================================\n";
    std::cout << "  H(curl) Jump-Coefficient Multigrid Benchmark\n";
    std::cout << "=================================================\n";
    std::cout << " Mesh File:       " << opts.mesh_file << "\n";
    std::cout << " Output File:     " << opts.output_file << "\n";
    std::cout << " B-Conditions:    " << opts.bcond << "\n";
    std::cout << " Cycle Type:      "
              << (cycle_type == CycleType::VCYCLE ? "V-Cycle" : "W-Cycle")
              << "\n";
    std::cout << " Lumping:         " << opts.lumping << "\n";
    std::cout << " Jump Value:      " << opts.jump_value << "\n";
    std::cout << " Inner Cube Size: " << opts.inner_cube_size << "\n";
    std::cout << " Max Refinement:  " << opts.max_ref << "\n";
    std::cout << " GMRES Runs:      " << opts.num_runs << "\n";
    std::cout << "=================================================\n\n";

    {
        std::ofstream ofs(opts.output_file, std::ios::out | std::ios::trunc);
    }

    mfem::Mesh mfem_mesh(opts.mesh_file.c_str());
    auto       mesh = std::make_shared<Mesh>(std::move(mfem_mesh));

    const mfem::real_t lo = 0.5 - 0.5 * opts.inner_cube_size;
    const mfem::real_t hi = 0.5 + 0.5 * opts.inner_cube_size;

    auto scalar_jump = std::make_shared<mfem::FunctionCoefficient>(
        [lo, hi, jump = opts.jump_value](const mfem::Vector& x) {
            return (x[0] >= lo && x[0] <= hi && x[1] >= lo && x[1] <= hi &&
                    x[2] >= lo && x[2] <= hi)
                       ? jump
                       : 1.0;
        });

    auto matrix_jump = std::make_shared<mfem::MatrixFunctionCoefficient>(
        3, [lo, hi, jump = opts.jump_value](const mfem::Vector& x,
                                              mfem::DenseMatrix& K) {
            const mfem::real_t value =
                (x[0] >= lo && x[0] <= hi && x[1] >= lo && x[1] <= hi &&
                 x[2] >= lo && x[2] <= hi)
                    ? jump
                    : 1.0;
            K.SetSize(3);
            K = 0.0;
            K(0, 0) = K(1, 1) = K(2, 2) = value;
        });

    ScalarMassCoefficientArray scalar_coeffs{};
    MatrixMassCoefficientArray matrix_coeffs{};
    scalar_coeffs[0] = scalar_jump;
    scalar_coeffs[2] = scalar_jump;
    matrix_coeffs[1] = matrix_jump;

    Multigrid<HCurlLaplacianOperator, HCurlLaplacianSmoother, 0, 1> MG(
        OperatorMode::Galerkin);
    MG.setCycleType(cycle_type);
    MG.setSmoothingIterations(opts.pre_smooth, opts.post_smooth);
    MG.setIterativeMode(false);
    MG.addCoarseLevel(
        nullptr, mesh, bc, OperatorMode::Galerkin, lumping, scalar_coeffs,
        matrix_coeffs);

    std::cout << std::setw(6) << "Lvl" << std::right << std::setw(15)
              << "Unknowns" << std::right << std::setw(12) << "h"
              << std::right << std::setw(16) << "Spectral Radii"
              << std::right << std::setw(16) << "Avg GMRES Iters"
              << std::right << std::setw(14) << "MG Rate" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (int lvl = 0; lvl <= opts.max_ref; ++lvl)
    {
        auto op = MG.getFinestOperator();
        const BenchResult res = run_level(
            MG, lvl, opts.num_runs, opts.gmres_tol, opts.krylov_dim,
            opts.gmres_maxit, opts.mg_iters, opts.verbose, bc,
            OperatorMode::Galerkin);

        save_to_csv(opts.output_file, res);

        std::cout << std::left << std::setw(6) << res.refinements << std::right
                  << std::setw(15) << res.dofs << std::right << std::scientific
                  << std::setprecision(4) << std::setw(12) << res.mesh_width
                  << std::right << std::setw(16) << res.spectral_radii
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(16) << res.avg_gmres_iters << std::right
                  << std::setw(14) << res.mg_asymptotic_rate << "\n";

        if (lvl < opts.max_ref)
            MG.addRefinedLevel();
    }

    std::cout << std::string(80, '=') << "\n";
    return 0;
}
