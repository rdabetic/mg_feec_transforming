#include <boost/json/src.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "SpectraErrorOp.hpp"
#include "dirac_2d.hpp"
#include "dirac_3d.hpp"
#include "hcurl_laplacian.hpp"
#include "magnetostatics.hpp"
#include "mg.tpp"

namespace json = boost::json;
namespace po   = boost::program_options;

struct BenchResult
{
    int          refinements;
    mfem::real_t mesh_width;
    int          dofs;
    std::string  spectral_radii;
    mfem::real_t avg_iters;
};

/**
 * Appends results to CSV. Header is added only if the file is empty.
 */
void save_to_csv(const std::string& filename, const BenchResult& res)
{
    std::ifstream check(filename);
    bool          exists = (check.peek() != std::ifstream::traits_type::eof());
    check.close();

    std::ofstream file(filename, std::ios::app);
    if (!exists)
        file << "Refinements,mesh-width,DOFs,SpectralRadii,AvgIters\n";

    file << res.refinements << "," << std::scientific << std::setprecision(6)
         << res.mesh_width << "," << res.dofs << ",\"" << res.spectral_radii
         << "\"," << std::fixed << std::setprecision(2) << res.avg_iters
         << "\n";
}

template <typename MGType>
BenchResult run_analysis(
    MGType&                            MG,
    mfem::BlockVector&                 start_v,
    const int                          current_ref,
    const int                          num_eigs,
    const mfem::real_t                 eig_tol,
    const int                          runs,
    const mfem::real_t                 gmres_tol,
    const int                          krylov_dim,
    const int                          gmres_maxit,
    const int                          ignore_gmres_fail,
    std::function<void(mfem::Vector&)> post = [](mfem::Vector&) {})
{
    auto op = MG.getFinestOperator();

    // 1. Eigenvalues (DEC Mode)
    std::stringstream ss_csv;

    MG.setMode(OperatorMode::DEC);
    op->setMode(OperatorMode::DEC);
    auto [ew, ev] = errorOpEig(*op, MG, start_v, num_eigs, eig_tol, post);

    for (int i = 0; i < ew.size(); ++i)
    {
        mfem::real_t mag = std::abs(ew[i]);
        ss_csv << mag << (i == ew.size() - 1 ? "" : " ");
    }

    // 2. GMRES Benchmark (Galerkin Mode)
    MG.setMode(OperatorMode::Galerkin);
    op->setMode(OperatorMode::Galerkin);
    int total_iters = 0;
    for (int r = 0; r < runs; ++r)
    {
        mfem::BlockVector rhs(op->getBlockOffsets()),
            x                      = op->createBlockVector();
        mfem::BlockVector rhs_rand = op->getRandomVector(54321 + r);
        op->eliminateTrivialSolutionHarmonics(rhs_rand);
        op->applyFemOp(rhs_rand, rhs);
        x = 0.0;

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

        if (!gmres.GetConverged())
            if (!ignore_gmres_fail)
                throw std::runtime_error(
                    "GMRES failed to converge after " +
                    std::to_string(gmres_maxit) + " iterations!");
            else
                std::cerr << "GMRES failed to converge after " +
                                 std::to_string(gmres_maxit) + " iterations!";
    }

    mfem::real_t avg =
        static_cast<mfem::real_t>(total_iters) / (runs > 0 ? runs : 1);

    return {
        current_ref, op->getMesh()->meshWidth(), op->Height(), ss_csv.str(),
        avg};
}

int main(int argc, char* argv[])
{
#ifdef MFEM_USE_OPENMP
    mfem::Device device("omp");
    device.Print(std::cout);
#endif

    // Define option storage with standard hardcoded defaults
    std::string config_path;
    std::string problem_str     = "mag";
    std::string mesh_file_str   = "cube.msh";
    std::string output_file_str = "results.csv";
    std::string bcond_str       = "essential";
    std::string lumping_str     = "rowsum";
    std::string cycle_str       = "v";

    int max_ref           = 2;
    int num_eigs          = 1;
    int krylov_dim        = 20;
    int runs              = 8;
    int gmres_maxit       = 1000;
    int ignore_gmres_fail = 0;
    int mg_iters          = 16;
    int verbose           = 1;
    int pre_smooth_iters  = 1;
    int post_smooth_iters = 1;

    double eig_tol   = 1e-2;
    double gmres_tol = 1e-6;

    if (typeid(mfem::real_t) == typeid(float))
        gmres_tol = 1e-4;

    try
    {
        // Define clean executable interface layout via Boost
        po::options_description desc("Allowed options");
        desc.add_options()("help,h", "Produce help message")(
            "config", po::value<std::string>(), "Path to JSON config file")(
            "problem,p", po::value<std::string>(),
            "mag, hcurl, dirac2d, dirac3d")(
            "mesh,m", po::value<std::string>(), "Mesh file")(
            "output,o", po::value<std::string>(), "CSV output file")(
            "max-ref,r", po::value<int>(), "Max refinements")(
            "num-eigs", po::value<int>(), "No. eigenvalues")(
            "eigtol", po::value<double>(), "Tolerance for eigenvalues")(
            "bcond", po::value<std::string>(), "essential, natural")(
            "lumping,l", po::value<std::string>(),
            "none, rowsum, diagonal, barycentric")(
            "cycle,c", po::value<std::string>(), "v, w")(
            "krylov,k", po::value<int>(), "GMRES Krylov dimension")(
            "num-runs,n", po::value<int>(), "No. GMRES runs")(
            "tol,t", po::value<double>(), "GMRES relative tolerance")(
            "maxit,N", po::value<int>(), "GMRES maxit")(
            "ignore,i", po::value<int>(), "Continue if GMRES fails")(
            "mg-iters,M", po::value<int>(),
            "Number of MG iterations to apply in residual experiment")(
            "verbose,V", po::value<int>(), "Verbose/debug output")(
            "pre-smooth", po::value<int>(),
            "Number of pre-smoothing iterations")(
            "post-smooth", po::value<int>(),
            "Number of post-smoothing iterations");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << desc << "\n";
            return 0;
        }

        // 1. JSON Configuration Fallback Parsing
        if (vm.count("config"))
        {
            config_path = vm["config"].as<std::string>();
            std::ifstream f(config_path);
            if (!f.is_open())
                throw std::runtime_error(
                    "❌ FATAL ERROR: Cannot open config file: " + config_path);

            std::string content(
                (std::istreambuf_iterator<char>(f)),
                std::istreambuf_iterator<char>());
            f.close();

            auto obj = json::parse(content).as_object();

            auto get_str = [&](const char* key) -> std::string
            {
                auto it = obj.find(key);
                if (it != obj.end() && it->value().is_string())
                    return it->value().as_string().c_str();
                return "";
            };

            auto get_int = [&](const char* key, int def) -> int
            {
                auto it = obj.find(key);
                if (it != obj.end() && it->value().is_int64())
                    return static_cast<int>(it->value().as_int64());
                return def;
            };

            auto get_double = [&](const char* key, double def) -> double
            {
                auto it = obj.find(key);
                if (it != obj.end() && it->value().is_double())
                    return it->value().as_double();
                return def;
            };

            auto get_bool = [&](const char* key, bool def) -> bool
            {
                auto it = obj.find(key);
                if (it != obj.end() && it->value().is_bool())
                    return it->value().as_bool();
                return def;
            };

            std::string val;
            if ((val = get_str("problem")).size())
                problem_str = val;
            if ((val = get_str("mesh_file")).size())
                mesh_file_str = val;
            if ((val = get_str("output_file")).size())
                output_file_str = val;
            if ((val = get_str("lumping")).size())
                lumping_str = val;
            if ((val = get_str("bcond")).size())
                bcond_str = val;
            if ((val = get_str("cycle")).size())
                cycle_str = val;

            max_ref           = get_int("max_ref", max_ref);
            num_eigs          = get_int("num_eigs", num_eigs);
            krylov_dim        = get_int("krylov_dim", krylov_dim);
            runs              = get_int("runs", runs);
            gmres_maxit       = get_int("gmres_maxit", gmres_maxit);
            ignore_gmres_fail = get_int("ignore_gmres_fail", ignore_gmres_fail);
            mg_iters          = get_int("mg_iters", mg_iters);
            verbose           = get_int("verbose", verbose);
            pre_smooth_iters  = get_int("pre_smooth_iters", pre_smooth_iters);
            post_smooth_iters = get_int("post_smooth_iters", post_smooth_iters);

            eig_tol   = get_double("eig_tol", eig_tol);
            gmres_tol = get_double("gmres_tol", gmres_tol);

            if (get_bool("ignore_gmres_fail", false))
                ignore_gmres_fail = 1;
            if (get_bool("verbose", true))
                verbose = 1;

            std::cout << "Loaded config: " << config_path << "\n";
        }

        // 2. Direct CLI Override Parsing
        if (vm.count("problem"))
            problem_str = vm["problem"].as<std::string>();
        if (vm.count("mesh"))
            mesh_file_str = vm["mesh"].as<std::string>();
        if (vm.count("output"))
            output_file_str = vm["output"].as<std::string>();
        if (vm.count("max-ref"))
            max_ref = vm["max-ref"].as<int>();
        if (vm.count("num-eigs"))
            num_eigs = vm["num-eigs"].as<int>();
        if (vm.count("eigtol"))
            eig_tol = vm["eigtol"].as<double>();
        if (vm.count("bcond"))
            bcond_str = vm["bcond"].as<std::string>();
        if (vm.count("lumping"))
            lumping_str = vm["lumping"].as<std::string>();
        if (vm.count("cycle"))
            cycle_str = vm["cycle"].as<std::string>();
        if (vm.count("krylov"))
            krylov_dim = vm["krylov"].as<int>();
        if (vm.count("num-runs"))
            runs = vm["num-runs"].as<int>();
        if (vm.count("tol"))
            gmres_tol = vm["tol"].as<double>();
        if (vm.count("maxit"))
            gmres_maxit = vm["maxit"].as<int>();
        if (vm.count("ignore"))
            ignore_gmres_fail = vm["ignore"].as<int>();
        if (vm.count("mg-iters"))
            mg_iters = vm["mg-iters"].as<int>();
        if (vm.count("verbose"))
            verbose = vm["verbose"].as<int>();
        if (vm.count("pre-smooth"))
            pre_smooth_iters = vm["pre-smooth"].as<int>();
        if (vm.count("post-smooth"))
            post_smooth_iters = vm["post-smooth"].as<int>();
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(
            "❌ FATAL ERROR: Arguments or config parsing failed: " +
            std::string(e.what()));
    }

    std::cout << "=================================================\n";
    std::cout << "  Multigrid Convergence Test Configuration\n";
    std::cout << "=================================================\n";
    std::cout << " Problem Type:    " << problem_str << "\n";
    std::cout << " Mesh File:       " << mesh_file_str << "\n";
    std::cout << " Output File:     " << output_file_str << "\n";
    std::cout << " MG Cycle Type:   "
              << (cycle_str == "v" ? "V-Cycle" : "W-Cycle") << "\n";
    std::cout << " Mass Lumping:    " << lumping_str << "\n";
    std::cout << " B-Conditions:    " << bcond_str << "\n";
    std::cout << " Max Refinement:  " << max_ref << "\n";
    std::cout << " Eigenvalues:     " << num_eigs << " (Tol: " << eig_tol
              << ")\n";
    std::cout << " GMRES Benchmark: " << runs << " runs (Tol: " << gmres_tol
              << ", K-Dim: " << krylov_dim << ")\n";
    std::cout << "=================================================\n\n";

    // Overwrite the file once at the start
    {
        std::ofstream ofs(output_file_str, std::ios::out | std::ios::trunc);
    }

    mfem::Mesh mfem_mesh(mesh_file_str, 1, 1);
    auto       mesh = std::make_shared<Mesh>(std::move(mfem_mesh));

    MassLumping                                               lumping;
    static const std::unordered_map<std::string, MassLumping> lumping_map = {
        {"none", MassLumping::None},
        {"rowsum", MassLumping::RowSum},
        {"barycentric", MassLumping::Barycentric},
        {"scaledid", MassLumping::ScaledId}};

    auto it_lump = lumping_map.find(lumping_str);
    if (it_lump != lumping_map.end())
        lumping = it_lump->second;
    else
        throw std::invalid_argument(
            "❌ FATAL ERROR: Invalid lumping type specified: '" + lumping_str +
            "'");

    BCond bc;
    if (bcond_str == "essential")
        bc = BCond::Essential;
    else if (bcond_str == "natural")
        bc = BCond::Natural;
    else
        throw std::invalid_argument(
            "❌ FATAL ERROR: BCond must be 'natural' or 'essential'. Received: "
            "'" +
            bcond_str + "'");

    CycleType cycle_type;
    if (cycle_str == "v")
        cycle_type = CycleType::VCYCLE;
    else if (cycle_str == "w")
        cycle_type = CycleType::WCYCLE;
    else
        throw std::invalid_argument(
            "❌ FATAL ERROR: Invalid cycle type specified: '" + cycle_str +
            "'. Use 'v' or 'w'.");

    auto setup_and_run = [&](auto& MG)
    {
        MG.setCycleType(cycle_type);
        MG.setSmoothingIterations(pre_smooth_iters, post_smooth_iters);
        MG.addCoarseLevel(nullptr, mesh, bc, OperatorMode::Galerkin, lumping);
        for (int r = 1; r <= max_ref; ++r)
        {
            MG.addRefinedLevel();
            auto op      = MG.getFinestOperator();
            auto start_v = op->getRandomVector(42);

            auto post = [&](mfem::Vector& x)
            {
                mfem::BlockVector bx(x.GetData(), op->getBlockOffsets());

                op->eliminateTrivialSolutionHarmonics(bx);
                if (bc == BCond::Essential)
                    op->eliminateBC(bx);
            };

            BenchResult bench_res = run_analysis(
                MG, start_v, r, num_eigs, eig_tol, runs, gmres_tol, krylov_dim,
                gmres_maxit, ignore_gmres_fail, post);

            save_to_csv(output_file_str, bench_res);

            std::cout
                << "======================================================\n";
            std::cout << "  Refinement Level: " << r << "\n";
            std::cout
                << "======================================================\n";
            std::cout << " Mesh width (h):  " << std::scientific
                      << std::setprecision(4) << bench_res.mesh_width << "\n";
            std::cout << " Total DOFs:      " << bench_res.dofs << "\n";
            std::cout << " Spectral Radii:  " << bench_res.spectral_radii
                      << "\n";
            std::cout << " GMRES Avg Iters: " << std::fixed
                      << std::setprecision(2) << bench_res.avg_iters << "\n";

            if (mg_iters > 0)
            {
                std::cout << "-------------------------------------------------"
                             "-----\n";
                std::cout << "  MG RESIDUAL DECAY\n";
                std::cout << "-------------------------------------------------"
                             "-----\n";

                MG.setMode(OperatorMode::DEC);
                MG.setIterativeMode(true);
                op->setMode(OperatorMode::DEC);

                mfem::BlockVector rhs(op->getBlockOffsets()),
                    x                      = op->createBlockVector();
                mfem::BlockVector rhs_rand = op->getRandomVector(12345 + r);
                op->eliminateTrivialSolutionHarmonics(rhs_rand);
                op->applyDecOp(rhs_rand, rhs);
                x = op->getRandomVector(54321 + r);

                double initial_res_norm = rhs.Norml2();
                if (initial_res_norm == 0.0)
                    initial_res_norm = 1.0;

                if (verbose)
                {
                    std::cout << " " << std::left << std::setw(8) << "Iter"
                              << std::setw(24) << "Rel. Residual"
                              << "Reduction Factor\n";
                    std::cout << "---------------------------------------------"
                                 "---------\n";
                }

                mfem::BlockVector   res       = op->createBlockVector();
                double              prev_norm = -1.0;
                std::vector<double> norms;

                for (int it = 0; it < mg_iters; ++it)
                {
                    MG.Mult(rhs, x);
                    op->Mult(x, res);
                    res -= rhs;

                    double nrm = res.Norml2() / initial_res_norm;
                    norms.push_back(nrm);
                    double red = (prev_norm > 0.0) ? nrm / prev_norm : 0.0;

                    if (verbose)
                    {
                        std::cout << " " << std::left << std::setw(8) << it
                                  << std::scientific << std::setprecision(6)
                                  << std::setw(24) << nrm;
                        if (prev_norm > 0.0)
                        {
                            std::cout << std::fixed << std::setprecision(4)
                                      << red;
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
                {
                    std::cout << "---------------------------------------------"
                                 "---------\n";
                }

                int    k       = std::min(4, (int) norms.size() / 2);
                int    start   = std::max(0, (int) norms.size() - k);
                double log_sum = 0.0;
                int    count   = 0;
                for (int i = start; i + 1 < (int) norms.size(); ++i)
                {
                    if (norms[i] > 1e-15 && norms[i + 1] > 1e-15)
                    {
                        log_sum += std::log(norms[i + 1] / norms[i]);
                        ++count;
                    }
                }
                double geom_mean_red = 1.0;
                if (count > 0)
                    geom_mean_red = std::exp(log_sum / count);

                if (geom_mean_red > 1.0)
                {
                    throw std::runtime_error(
                        "Geometric mean reduction factor > 1.0 indicates "
                        "divergence or noise "
                        "floor issues in the residual decay experiment.");
                }

                std::cout << " Iterations:      " << mg_iters << "\n";
                std::cout << " Final Rel. Res.: " << std::scientific
                          << std::setprecision(6) << norms.back() << "\n";
                std::cout << " Asymptotic Rate: " << std::fixed
                          << std::setprecision(6) << geom_mean_red
                          << " (Geom. Mean, last " << k << " iters)\n";

                MG.setMode(OperatorMode::Galerkin);
                op->setMode(OperatorMode::Galerkin);
                MG.setIterativeMode(false);
            }
            std::cout
                << "======================================================\n\n";
        }
    };

    if (problem_str == "mag")
    {
        using namespace Magnetostatics;
        Multigrid<MagOperator, MagSmoother, 0, 1> MG;
        setup_and_run(MG);
    }
    else if (problem_str == "hcurl")
    {
        using namespace HCurlLaplacian;
        Multigrid<HCurlLaplacianOperator, HCurlLaplacianSmoother, 0, 1> MG;
        setup_and_run(MG);
    }
    else if (problem_str == "dirac2d")
    {
        using namespace Dirac2D;
        Multigrid<DiracOperator2D, DiracSmoother2D, 0, 1, 2> MG;
        setup_and_run(MG);
    }
    else if (problem_str == "dirac3d")
    {
        using namespace Dirac3D;
        Multigrid<DiracOperator3D, DiracSmoother3D, 0, 1, 2, 3> MG;
        setup_and_run(MG);
    }
    else
    {
        throw std::invalid_argument(
            "❌ FATAL ERROR: Unknown problem type '" + problem_str +
            "' specified!");
    }

    return 0;
}