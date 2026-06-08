#!/usr/bin/env python3
"""
Generate JSON config files for cvg into a flat directory.
All simulation sweep parameters are exposed as global constants at the top for easy tweaking.
Generates portable relative paths so configs can be moved across machines seamlessly.
"""

import json
import argparse
import sys
import shutil
from pathlib import Path

# ==============================================================================
# GLOBAL CONFIGURATION OPTIONS - EDIT THESE TO TWEAK YOUR MATRIX SWEEP
# ==============================================================================

# List of mesh filenames (without extension) located in meshes/2d/ and meshes/3d/
MESHES_2D = ["circle", "square_centered", "tria", 
             "L", "reentrant", "square_hole", 
             "two_holes"]
MESHES_3D = ["cube", "ball", "tetra", 
             "corner", "corner_structured", "hole_void"]

# Reduced subset of meshes strictly used for --test mode
TEST_MESHES_2D = ["L", "two_holes"]
TEST_MESHES_3D = ["hole_void"]

# Refinement levels, baseline settings, run counts, and SLURM footprints
STANDARD_SETTINGS = {
    "ref_2d": 6,
    "ref_3d": 5,
    "runs": 8,
    "mem_2d": "16G",
    "time_2d": "00:45:00",
    "mem_3d": "64G",
    "time_3d": "02:00:00"
}

# Settings used when running with the --test validation flag
TEST_SETTINGS = {
    "ref_2d": 2,
    "ref_3d": 1,   # Set to 0 to completely skip 3D sweeps in test mode
    "runs": 1,
    "mem_2d": "1G",       
    "time_2d": "00:05:00", 
    "mem_3d": "4G",
    "time_3d": "00:05:00"
}

# ------------------------------------------------------------------------------
# ADDITIONAL SMOOTHING ITERATION SWEEP
# ------------------------------------------------------------------------------
# If True (and not in --test mode), the script will generate the normal config 
# for the target mesh/problem, AND IN ADDITION generate a sweep of extra configs 
# with pre_smooth_iters=0 and post_smooth_iters=k.
ENABLE_SMOOTH_SWEEP = True
SWEEP_TARGET_MESH   = "corner"
SWEEP_TARGET_PROB   = "mag"
SWEEP_K_VALUES      = [1, 2, 3, 4]

# ------------------------------------------------------------------------------
# TOPOLOGICAL CONFIGURATIONS & EIGENVALUE LAMBDAS
# ------------------------------------------------------------------------------
# Explicitly map each mesh to its Betti numbers: (b0, b1, b2)
# b0 = connected components, b1 = tunnels/holes, b2 = cavities/voids
MESH_BETTI_NUMBERS = {
    # 2D Meshes (b2 is always 0)
    "circle":           (1, 0, 0),
    "square_centered":  (1, 0, 0),
    "tria":             (1, 0, 0),
    "L":                (1, 0, 0),
    "reentrant":        (1, 0, 0),
    "square_hole":      (1, 1, 0),
    "two_holes":        (1, 2, 0),
    
    # 3D Meshes
    "cube":              (1, 0, 0),
    "ball":              (1, 0, 0),
    "tetra":             (1, 0, 0),
    "corner":            (1, 0, 0),
    "corner_structured": (1, 0, 0),
    "hole_void":         (1, 1, 1),
}

# Define a lambda function for each problem type to calculate 'nev' dynamically.
# The lambda accepts 'b' (a tuple of (b0, b1, b2)) AND 'dim' ("2d" or "3d").
PROBLEM_NEV_FUNCTIONS = {
    "hcurl":   lambda b, dim: b[1] + b[0] if dim == "2d" else b[2] + b[0],
    "mag":     lambda b, dim: b[1] + b[0] if dim == "2d" else b[2] + b[0],
    "dirac2d": lambda b, dim: sum(b),
    "dirac3d": lambda b, dim: sum(b),
}

# ------------------------------------------------------------------------------

# Mass matrix lumping schemes and multigrid structure patterns
LUMPING_MODES = ["rowsum", "barycentric", "scaledid"]
CYCLE_MODES   = ["v", "w"]

# Problem definitions: (problem_identifier, output_subdir, strict_dim_matching, allowed_dimensions)
PROBLEMS = [
    ("dirac2d", "dirac", False, None),
    ("dirac3d", "dirac", False, None),
    ("hcurl",   "veclap", True,  ["2d", "3d"]),
    ("mag",     "mag",   True,  ["2d", "3d"]),
]

# Base numeric conditions and solver metrics applied uniformly across configurations
COMMON_SOLVER_OPTS = {
    "eig_tol": 1e-2,
    "bcond": "essential",
    "krylov_dim": 20,
    "gmres_tol": 1e-6,
    "gmres_maxit": 256,
    "ignore_gmres_fail": False,
    "mg_iters": 16,
    "verbose": True,
    "pre_smooth_iters": 1,
    "post_smooth_iters": 1,
}

# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate flat JSON configs for cvg")
    parser.add_argument("--out-dir", default="configs", help="Output directory for JSONs (default: configs)")
    parser.add_argument("--test", action="store_true", help="Generate a small, fast subset with low resource limits for pipeline testing")
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    
    if output_dir.exists():
        print(f"Cleaning existing directory: {output_dir}/")
        shutil.rmtree(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = TEST_SETTINGS if args.test else STANDARD_SETTINGS

    ref_2d = settings["ref_2d"]
    ref_3d = settings["ref_3d"]
    runs   = settings["runs"]

    active_meshes_2d = TEST_MESHES_2D if args.test else MESHES_2D
    active_meshes_3d = TEST_MESHES_3D if args.test else MESHES_3D

    all_meshes = []
    if ref_2d > 0:
        all_meshes.extend([("2d", m, ref_2d) for m in active_meshes_2d])
    if ref_3d > 0:
        all_meshes.extend([("3d", m, ref_3d) for m in active_meshes_3d])

    count = 0
    sweep_count = 0
    
    for dim, mesh_name, ref in all_meshes:
        
        portable_mesh_path = f"meshes/{dim}/{mesh_name}.msh"

        for lump in LUMPING_MODES:
            for cycle in CYCLE_MODES:
                for prob_name, out_subdir, has_dim, dims in PROBLEMS:
                    
                    if has_dim:
                        if dim not in dims:
                            continue
                    else:
                        if (prob_name == "dirac2d" and dim == "3d") or \
                           (prob_name == "dirac3d" and dim == "2d"):
                            continue

                    betti = MESH_BETTI_NUMBERS.get(mesh_name, (1, 0, 0))
                    if prob_name in PROBLEM_NEV_FUNCTIONS:
                        current_nev = PROBLEM_NEV_FUNCTIONS[prob_name](betti, dim)
                    else:
                        current_nev = 1

                    mem_str = settings[f"mem_{dim}"]
                    time_str = settings[f"time_{dim}"]

                    # 1. ALWAYS GENERATE THE STANDARD CONFIG
                    out_path = f"out/{lump}/{cycle}/{out_subdir}/{dim}/{mesh_name}.csv"

                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    cfg = {
                        "problem": prob_name,
                        "mesh_file": portable_mesh_path,
                        "output_file": out_path,
                        "max_ref": ref,
                        "num_eigs": current_nev,
                        "runs": runs,
                        "lumping": lump,
                        "cycle": cycle,
                        "slurm_mem": mem_str,
                        "slurm_time": time_str,
                    }
                    cfg.update(COMMON_SOLVER_OPTS)
                    
                    cfg_filename = f"{prob_name}_{dim}_{mesh_name}_{lump}_{cycle}.json"
                    with open(output_dir / cfg_filename, "w") as f:
                        json.dump(cfg, f, indent=2)
                        f.write("\n")
                    count += 1

                    # 2. GENERATE EXTRA SWEEP CONFIGS IF TARGET MATCHES
                    if ENABLE_SMOOTH_SWEEP and not args.test:
                        if mesh_name == SWEEP_TARGET_MESH and prob_name == SWEEP_TARGET_PROB:
                            for k in SWEEP_K_VALUES:
                                out_path_k = f"out/{lump}/{cycle}/{out_subdir}/{dim}/{mesh_name}_k{k}.csv"
                                
                                Path(out_path_k).parent.mkdir(parents=True, exist_ok=True)

                                cfg_k = dict(cfg)
                                cfg_k["output_file"] = out_path_k
                                cfg_k["pre_smooth_iters"] = 0
                                cfg_k["post_smooth_iters"] = k
                                
                                cfg_filename_k = f"{prob_name}_{dim}_{mesh_name}_{lump}_{cycle}_k{k}.json"
                                with open(output_dir / cfg_filename_k, "w") as f:
                                    json.dump(cfg_k, f, indent=2)
                                    f.write("\n")
                                sweep_count += 1

    mode_label = "TEST" if args.test else "STANDARD"
    print(f"Generated {count} {mode_label} config files in {output_dir}/")
    if sweep_count > 0:
        print(f"  -> Plus {sweep_count} additional smoothing sweep configs for {SWEEP_TARGET_MESH} ({SWEEP_TARGET_PROB})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
