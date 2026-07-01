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
#include "magnetostatics.hpp"
#include "mesh.hpp"
#include "mfem.hpp"
#include "mg.tpp"
#include "operator.tpp"

namespace po = boost::program_options;
using namespace Magnetostatics;

namespace
{
struct CliOptions
{
    std::string mesh_file = "../../tests/cube.msh";
    std::string output_file = "mag_jump_residual_decay.txt";
    std::string bcond = "essential";
    std::string cycle = "w";
    std::string lumping = "rowsum";
    int         max_ref = 3;
    int         num_runs = 8;
    int         krylov_dim = 32;
    int         gmres_maxit = 256;
    int         mg_iters = 16;
    int         pre_smooth = 1;
    int         post_smooth = 1;
    int         verbose = 0;
    double      gmres_tol = 1e-6;
    double      jump_min = 1;
    double      jump_max = 1e4;
    int         jump_steps = 5;
    double      inner_cube_size = 0.5;
};

struct JumpResult
{
    double      jump_value;
    int         ref_level;
    std::string spectral_radii;
    mfem::real_t avg_gmres_iters;
    mfem::real_t mg_asymptotic_rate;
};

void print_and_save_table(const std::string& csv_filename,
                          const std::vector<JumpResult>& results)
{
    // Print to stdout
    std::cout << std::left << std::setw(14) << "Jump Value" << std::right
              << std::setw(12) << "Ref Level" << std::right
              << std::setw(16) << "Spectral Radii" << std::right
              << std::setw(16) << "Avg GMRES Iters" << std::right
              << std::setw(14) << "MG Rate" << "\n";
    std::cout << std::string(72, '-') << "\n";

    for (const auto& res : results)
    {
        std::cout << std::left << std::setw(14) << std::scientific
                  << std::setprecision(2) << res.jump_value << std::right
                  << std::setw(12) << res.ref_level << std::right
                  << std::setw(16) << res.spectral_radii << std::right
                  << std::fixed << std::setprecision(2) << std::setw(16)
                  << res.avg_gmres_iters << std::right << std::setw(14)
                  << res.mg_asymptotic_rate << "\n";
    }

    // Save to CSV
    std::ofstream file(csv_filename);
    file << "Jump,RefLevel,SpectralRadii,AvgGMRESIters,MGAsymptoticRate\n";
    for (const auto& res : results)
    {
        file << std::scientific << std::setprecision(6) << res.jump_value
             << "," << res.ref_level
             << ",\"" << res.spectral_radii << "\","
             << std::fixed << std::setprecision(2) << res.avg_gmres_iters
             << "," << std::scientific << std::setprecision(6)
             << res.mg_asymptotic_rate << "\n";
    }
}

CliOptions parse_options(int argc, char* argv[])
{
    CliOptions opts;

    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Show help message")(
        "mesh,m", po::value<std::string>(&opts.mesh_file)->default_value(opts.mesh_file),
        "Mesh file (default: unit cube mesh)")(
        "output,o", po::value<std::string>(&opts.output_file)->default_value(opts.output_file),
        "Output file for residual decay")(
        "bcond", po::value<std::string>(&opts.bcond)->default_value(opts.bcond),
        "essential, natural")(
        "cycle,c", po::value<std::string>(&opts.cycle)->default_value(opts.cycle),
        "v, w")(
        "lumping,l", po::value<std::string>(&opts.lumping)->default_value(opts.lumping),
        "rowsum, scaledid")(
        "max-ref,r", po::value<int>(&opts.max_ref)->default_value(opts.max_ref),
        "Maximum refinement level to test")(
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
        "jump-min", po::value<double>(&opts.jump_min)->default_value(opts.jump_min),
        "Minimum jump coefficient value")(
        "jump-max", po::value<double>(&opts.jump_max)->default_value(opts.jump_max),
        "Maximum jump coefficient value")(
        "jump-steps", po::value<int>(&opts.jump_steps)->default_value(opts.jump_steps),
        "Number of jump values to test (logarithmic spacing)")(
        "inner-size", po::value<double>(&opts.inner_cube_size)->default_value(opts.inner_cube_size),
        "Side length of the inner cube jump region")(
        "verbose,V", po::value<int>(&opts.verbose)->default_value(opts.verbose),
        "Verbose output");

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
JumpResult run_level(
    MGType&                     MG,
    const double                jump_value,
    const int                   ref_level,
    const int                   num_runs,
    const mfem::real_t          gmres_tol,
    const int                   krylov_dim,
    const int                   gmres_maxit,
    const int                   mg_iters,
    const int                   verbose,
    const BCond                 bc,
    const OperatorMode          mode,
    const int                   num_eigs = 1,
    const mfem::real_t          eig_tol = 1e-2,
    std::ofstream* residual_out = nullptr)
{
    auto op = MG.getFinestOperator();

    // Compute spectral radii (DEC mode)
    std::stringstream ss_csv;
    MG.setMode(OperatorMode::DEC);
    op->setMode(OperatorMode::DEC);
    auto start_v = op->getRandomVector(42 + ref_level);
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
        mfem::BlockVector x_true = op->getRandomVector(12345 + 17 * ref_level + r);
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
        mfem::BlockVector x = op->getRandomVector(54321 + ref_level);
        mfem::BlockVector rhs_rand = op->getRandomVector(12345 + ref_level);
        op->eliminateTrivialSolutionHarmonics(rhs_rand);
        op->applyDecOp(rhs_rand, rhs);

        mfem::BlockVector res = op->createBlockVector();

        op->Mult(x, res);
        res -= rhs;
        const mfem::real_t initial_norm = std::max<mfem::real_t>(rhs.Norml2(), 1e-16);

        std::vector<mfem::real_t> norms;
        norms.reserve(mg_iters);
        mfem::real_t prev_norm = -1.0;

        // Write header to file if provided
        if (residual_out)
        {
            *residual_out << "Jump Value: " << jump_value 
                          << " | Ref Level: " << ref_level << "\n";
            *residual_out << std::left << std::setw(8) << "Iter"
                          << std::setw(24) << "Rel. Residual"
                          << "Reduction Factor\n";
            *residual_out << std::string(55, '-') << "\n";
        }

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

            const mfem::real_t nrm = res.Norml2() / initial_norm;
            norms.push_back(nrm);
            const mfem::real_t red = (prev_norm > 0.0) ? nrm / prev_norm : 0.0;

            if (residual_out)
            {
                *residual_out << " " << std::left << std::setw(8) << it
                              << std::scientific << std::setprecision(6)
                              << std::setw(24) << nrm;
                if (prev_norm > 0.0)
                {
                    *residual_out << std::fixed << std::setprecision(4) << red;
                }
                else
                {
                    *residual_out << "-----";
                }
                *residual_out << "\n";
            }

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

        if (residual_out)
            *residual_out << std::string(55, '-') << "\n\n";

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
        jump_value, ref_level, ss_csv.str(), avg_iters, mg_rate
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
    std::cout << "  Magnetostatics 2-Form Jump Multigrid Benchmark\n";
    std::cout << "=================================================\n";
    std::cout << " Mesh File:       " << opts.mesh_file << "\n";
    std::cout << " Output File:     " << opts.output_file << "\n";
    std::cout << " B-Conditions:    " << opts.bcond << "\n";
    std::cout << " Cycle Type:      "
              << (cycle_type == CycleType::VCYCLE ? "V-Cycle" : "W-Cycle")
              << "\n";
    std::cout << " Lumping:         " << opts.lumping << "\n";
    std::cout << " Max Refinement:  " << opts.max_ref << "\n";
    std::cout << " Jump Range:      [" << opts.jump_min << ", " << opts.jump_max
              << "] (" << opts.jump_steps << " steps)\n";
    std::cout << " Inner Cube Size: " << opts.inner_cube_size << "\n";
    std::cout << " GMRES Runs:      " << opts.num_runs << "\n";
    std::cout << "=================================================\n\n";

    // Generate logarithmically spaced jump values
    std::vector<double> jump_values;
    {
        const double log_min = std::log(opts.jump_min);
        const double log_max = std::log(opts.jump_max);
        const double step = (opts.jump_steps > 1)
                                ? (log_max - log_min) / (opts.jump_steps - 1)
                                : 0.0;
        for (int i = 0; i < opts.jump_steps; ++i)
        {
            jump_values.push_back(std::exp(log_min + i * step));
        }
    }

    // Open residual decay output file
    std::ofstream residual_file(opts.output_file);

    // Results container
    std::vector<JumpResult> results;

    mfem::Mesh mfem_mesh(opts.mesh_file.c_str());
    auto       mesh = std::make_shared<Mesh>(std::move(mfem_mesh));

    const mfem::real_t lo = 0.5 - 0.5 * opts.inner_cube_size;
    const mfem::real_t hi = 0.5 + 0.5 * opts.inner_cube_size;

    // Iterate over jump values
    for (double jump_value : jump_values)
    {
        // Create the jumping matrix coefficient for the 2-form inner product
        auto matrix_jump = std::make_shared<mfem::MatrixFunctionCoefficient>(
            3, [lo, hi, jump = jump_value](const mfem::Vector& x, mfem::DenseMatrix& K) {
                const mfem::real_t value = 
                    (x[0] >= lo && x[0] <= hi && 
                     x[1] >= lo && x[1] <= hi && 
                     x[2] >= lo && x[2] <= hi) ? jump : 1.0;
                K.SetSize(3);
                K = 0.0;
                K(0,0) = K(1,1) = K(2,2) = value;
            });

        // Create matrix coefficient array with jump in 2-form (index 2)
        MatrixMassCoefficientArray matrix_coeffs = {};
        matrix_coeffs[2] = matrix_jump;

        Multigrid<MagOperator, MagSmoother, 0, 1> MG(
            OperatorMode::Galerkin);
        MG.setCycleType(cycle_type);
        MG.setSmoothingIterations(opts.pre_smooth, opts.post_smooth);
        MG.setIterativeMode(false);
        
        // Pass the matrix coefficient array to apply jump in 2-form inner product
        MG.addCoarseLevel(nullptr, mesh, bc, OperatorMode::Galerkin, lumping, 
                          ScalarMassCoefficientArray{}, matrix_coeffs);

        // Test across refinement levels for this jump value
        for (int r = 0; r <= opts.max_ref; ++r)
        {
            if (r > 0)
            {
                MG.addRefinedLevel();
            }

            // Run analysis at this jump value and refinement level
            const JumpResult bench_res =
                run_level(MG, jump_value, r, opts.num_runs,
                          opts.gmres_tol, opts.krylov_dim, opts.gmres_maxit,
                          opts.mg_iters, opts.verbose, bc, OperatorMode::Galerkin,
                          1, 1e-2, &residual_file);

            results.push_back(bench_res);
        }
    }

    residual_file.close();

    // Print and save results table
    std::string csv_output = opts.output_file;
    // Replace .txt with .csv if applicable
    const size_t dot_pos = csv_output.rfind('.');
    if (dot_pos != std::string::npos)
        csv_output = csv_output.substr(0, dot_pos) + "_results.csv";
    else
        csv_output += "_results.csv";

    std::cout << "\n=================================================\n";
    std::cout << "  Jump Coefficient Study Results\n";
    std::cout << "=================================================\n\n";
    print_and_save_table(csv_output, results);

    std::cout << "\n=================================================\n";
    std::cout << "Residual decay data saved to: " << opts.output_file << "\n";
    std::cout << "Results table saved to:      " << csv_output << "\n";
    std::cout << "=================================================\n";

    return 0;
}