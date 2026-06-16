import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio
import numpy as np
from pathlib import Path
from collections import Counter
import re
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import os

import pyvista as pv

# Suppress warnings that cause terminal whitespace and break tqdm formatting
warnings.filterwarnings("ignore")

# Force PyVista to run headless so it doesn't try to open windows
pv.OFF_SCREEN = True

# ==========================================
#              HELPER FUNCTIONS
# ==========================================

def get_title_from_path(path):
    # Lowercase all parts for highly robust substring matching
    parts = [str(p).lower() for p in path.parts]
    
    prob_str = "Unknown Problem"
    if any("veclap" in p or "vector" in p or "laplac" in p for p in parts): 
        prob_str = "1-form Hodge Laplacian"
    elif any("dirac" in p for p in parts): 
        prob_str = "Dirac"
    elif any("mag" in p for p in parts): 
        prob_str = "Magnetostatics"
    
    cycle_str = "Unknown Cycle"
    # Exact match for "v" or "w", or substring match for "vcycle"/"w-cycle"
    if any(p == "v" or "vcycle" in p or "v-cycle" in p for p in parts): 
        cycle_str = "V-Cycle"
    elif any(p == "w" or "wcycle" in p or "w-cycle" in p for p in parts): 
        cycle_str = "W-Cycle"
    
    lump_str = "Unknown Lumping"
    if any("barycentric" in p or "bary" in p for p in parts): 
        lump_str = "Barycentric"
    elif any("diagonal" in p or "rowsum" in p or "row_sum" in p for p in parts): 
        lump_str = "Row Sum"
    elif any("scaledid" in p or "scaled" in p for p in parts): 
        lump_str = "Scaled Id"
    
    return f"{prob_str}: {cycle_str} ({lump_str})"

def clean_dataframe_columns(df):
    df.columns = df.columns.astype(str).str.replace('#', '').str.strip()
    return df

def parse_ugly_values(val):
    if pd.isna(val): return [np.nan]
    if isinstance(val, (int, float)): return [float(val)]
    val_str = str(val).strip()
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val_str)
    if not numbers: return [np.nan]
    if '(' in val_str and ')' in val_str and len(numbers) == 2 and '[' not in val_str:
        real, imag = float(numbers[0]), float(numbers[1])
        return [np.sqrt(real**2 + imag**2)]
    return [float(x) for x in numbers]

def get_dominant_eigenvalue_and_k(vals):
    if not isinstance(vals, list) or not vals:
        return np.nan, None
    sorted_vals = sorted([v for v in vals if not np.isnan(v)], reverse=True)
    for i, v in enumerate(sorted_vals):
        if not np.isclose(v, 1.0, atol=1e-3):
            return v, i + 1  
    return np.nan, None

def select_columns_dec(columns):
    target_x = ["level", "leve", "refinement", "ref"]
    y_keys = ["lambda", "error", "err", "dec", "eig", "val", "norm", "l2", "h1", "diff", "spectral", "radius", "rho", "laplace", "vector"]
    found_x, found_y = None, []
    for c in columns:
        if found_x is None and any(x in c.lower() for x in target_x): found_x = c
        elif any(y in c.lower() for y in y_keys): found_y.append(c)
    return found_x, list(dict.fromkeys(found_y))

def select_columns_gmres(columns):
    target_x = ["level", "leve", "refinement", "ref"]
    y_keys = ["steps", "step", "iter", "iterations", "res", "gmres", "count", "ksp"]
    found_x, found_y = None, None
    for c in columns:
        if found_x is None and any(x in c.lower() for x in target_x): found_x = c
        elif found_y is None and any(y in c.lower() for y in y_keys): found_y = c
    return found_x, found_y

def extract_series(file_path, plot_type="dec"):
    try:
        df = clean_dataframe_columns(pd.read_csv(file_path, skipinitialspace=True, sep=r'\s*,\s*|;', engine='python'))
        if plot_type == "dec":
            x_col, y_cols = select_columns_dec(df.columns)
            if not x_col or not y_cols: return None, None, None
            df = df.sort_values(by=x_col)
            for y_col in y_cols:
                parsed = df[y_col].apply(parse_ugly_values)
                y_srs, k_srs = [], []
                for p in parsed:
                    v, k = get_dominant_eigenvalue_and_k(p)
                    y_srs.append(v)
                    if k is not None: k_srs.append(k)
                y_srs = pd.Series(y_srs)
                if not y_srs.isna().all():
                    k_mode = Counter(k_srs).most_common(1)[0][0] if k_srs else "?"
                    return df[x_col].astype(int), y_srs, k_mode
            return None, None, None
        else:
            x_col, y_col = select_columns_gmres(df.columns)
            if not x_col or not y_col: return None, None, None
            df = df.sort_values(by=x_col)
            y_srs = df[y_col].apply(parse_ugly_values).apply(lambda x: x[0] if x and not pd.isna(x[0]) else np.nan)
            return df[x_col].astype(int), y_srs, None
    except Exception:
        return None, None, None

def add_sorted_legend(ax_or_fig, handles, labels, **kwargs):
    if not labels: return
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0].lower())
    sorted_labels, sorted_handles = zip(*sorted_pairs)
    ax_or_fig.legend(sorted_handles, sorted_labels, **kwargs)

# ==========================================
#          THE "GHOST MESH" RENDERER
# ==========================================

def render_mesh_image(msh_file, output_path):
    try:
        mesh = pv.read(msh_file)
        if mesh.n_points == 0:
            file_name = getattr(msh_file, 'name', str(msh_file))
            print(f"Skipping {file_name}: Mesh contains no points.")
            return False

        surf = mesh.extract_surface()
        
        plotter = pv.Plotter(off_screen=True, window_size=[2000, 2000])
        plotter.set_background("white")
        
        bounds = mesh.bounds
        is_3d = (bounds[5] - bounds[4]) > 1e-5

        if not is_3d:
            # 2D MESH RENDERING
            plotter.add_mesh(
                surf, 
                color="white",             
                show_edges=True,           
                edge_color="black",        
                line_width=1.0,            
                lighting=False             
            )
            outlines_2d = surf.extract_feature_edges(
                boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
            )
            if outlines_2d.n_cells > 0:
                plotter.add_mesh(outlines_2d, color="black", line_width=3.5, lighting=False)
            plotter.view_xy()
        else:
            # 3D MESH RENDERING
            plotter.add_mesh(
                surf, 
                color="whitesmoke",             
                show_edges=True,          
                edge_color="#d0d0d0",      
                line_width=0.5,            
                opacity=0.3,               
                culling="back",            
                lighting=True,             
                smooth_shading=True
            )
            outlines_3d = surf.extract_feature_edges(
                feature_angle=45, boundary_edges=True, feature_edges=True, manifold_edges=False
            )
            if outlines_3d.n_cells > 0:
                plotter.add_mesh(outlines_3d, color="black", line_width=2.5, lighting=False)

            plotter.camera_position = 'xy'
            plotter.camera.azimuth = -35
            plotter.camera.elevation = 25

        plotter.reset_camera()
        
        out_path = Path(output_path).with_suffix('.png')
        plotter.screenshot(str(out_path))
        plotter.close()
        return True
        
    except Exception as e:
        file_name = getattr(msh_file, 'name', str(msh_file))
        print(f"Error rendering {file_name}: {e}")
        return False

# ==========================================
#              PLOTTING WORKERS
# ==========================================

def plot_mesh_standalone(msh_file, target_dir):
    """Saves a standalone mesh image."""
    output_path = target_dir / f"{msh_file.stem}.png"
    render_mesh_image(msh_file, output_path)

def plot_single_mesh_1x2(csv_file, msh_file, target_dir, style_dict):
    """Combines PyVista mesh render with Matplotlib line plots."""
    try:
        x_dec, y_dec, k = extract_series(csv_file, "dec")
        x_gm, y_gm, _ = extract_series(csv_file, "gmres")
        if (x_dec is None or y_dec.isna().all()) and (x_gm is None or y_gm.isna().all()): return

        fig = plt.figure(figsize=(14, 6.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.1])
        
        ax0 = fig.add_subplot(gs[0])
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            
        success = render_mesh_image(msh_file, tmp_path)
        
        if success:
            img = plt.imread(tmp_path)
            ax0.imshow(img)
            ax0.axis('off')
        else:
            ax0.text(0.5, 0.5, "Mesh Render Failed", ha='center', va='center')
            ax0.axis('off')
            
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        
        ax1 = fig.add_subplot(gs[1])
        ax2 = ax1.twinx()
        lines, labels = [], []
        
        if x_dec is not None and not y_dec.isna().all():
            l1 = ax1.plot(x_dec, y_dec, color='tab:blue', marker='o', linestyle='-', linewidth=2.5, markersize=10, 
                          label=f"Spectral Radii ($n_\\mathrm{{{{ev}}}}={k}$)" if k is not None else "Spectral Radii")
            lines.extend(l1); labels.append(l1[0].get_label())
            ax1.set_ylabel(r"$|\lambda|$", fontsize=22, color='tab:blue') 
            ax1.set_ylim(0.0, 1.02)
            ax1.xaxis.get_major_locator().set_params(integer=True)
            
        if x_gm is not None and not y_gm.isna().all():
            l2 = ax2.plot(x_gm, y_gm, color='tab:red', marker='s', linestyle='--', linewidth=2.5, markersize=10, label="GMRES Steps")
            lines.extend(l2); labels.append(l2[0].get_label())
            ax2.set_ylabel("GMRES Steps", fontsize=22, color='tab:red') 
            bottom_gm, top_gm = ax2.get_ylim()
            ax2.set_ylim(0, max(10, top_gm * 1.1))

        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax2.tick_params(axis='y', which='major', labelsize=18)

        ax1.set_title(get_title_from_path(csv_file.parent), fontsize=24, pad=15)
        ax1.set_xlabel("Refinement Level", fontsize=22)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        if lines:
            ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.28), fontsize=18, ncol=2)
        
        fig.subplots_adjust(wspace=0.1, left=0.02, right=0.88, top=0.85, bottom=0.35)
        fig.savefig(target_dir / f"{csv_file.stem}_combined_1x2.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error in 1x2 plot for {csv_file}: {e}")
        plt.close('all')

def plot_dec(directory_path, files, output_dir, style_dict):
    if not files: return
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for file_path in files:
        x, y, k = extract_series(file_path, "dec")
        if x is not None and not y.isna().all():
            c, m, ls = style_dict.get(file_path.stem, ('k', 'o', '-'))
            label = f"{file_path.stem} ($n_\\mathrm{{{{ev}}}}={k}$)" if k is not None else file_path.stem
            ax.plot(x, y, label=label, color=c, marker=m, linestyle=ls, linewidth=1.5, markersize=8)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(f"{get_title_from_path(directory_path)} - Spectral Radii", fontsize=16, pad=15)
    ax.set_xlabel("Refinement Level", fontsize=14)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.get_major_locator().set_params(integer=True)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        add_sorted_legend(fig, handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=14, ncol=4, frameon=True, borderpad=1, handlelength=3)
    fig.savefig(output_dir / "combined_plot_dec.pdf", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def plot_gmres(directory_path, files, output_dir, style_dict):
    if not files: return
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for file_path in files:
        x, y, _ = extract_series(file_path, "gmres")
        if x is not None and not y.isna().all():
            c, m, ls = style_dict.get(file_path.stem, ('k', 'o', '-'))
            ax.plot(x, y, label=file_path.stem, color=c, marker=m, linestyle=ls, linewidth=1.5, markersize=8)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(f"{get_title_from_path(directory_path)} - GMRES Iterations", fontsize=16, pad=15)
    ax.set_xlabel("Refinement Level", fontsize=14)
    ax.set_ylabel("Steps", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.get_major_locator().set_params(integer=True)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        add_sorted_legend(fig, handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=14, ncol=4, frameon=True, borderpad=1, handlelength=3)
    fig.savefig(output_dir / "combined_plot_gmres.pdf", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def plot_summary_2x2(prob_dir, csvs_2d, csvs_3d, output_dir, style_dict):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex='col')
    fig.suptitle(get_title_from_path(prob_dir), fontsize=18, y=0.98, fontweight='bold')
    
    def plot_on_ax(ax, files, plot_type):
        has_data = False
        for f in files:
            x, y, k = extract_series(f, plot_type)
            if x is not None and not y.isna().all():
                c, m, ls = style_dict.get(f.stem, ('k', 'o', '-'))
                label = f"{f.stem} ($n_\\mathrm{{{{ev}}}}={k}$)" if k is not None else f.stem
                ax.plot(x, y, label=label, color=c, marker=m, linestyle=ls, linewidth=1.5, markersize=7)
                has_data = True
        if has_data:
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.xaxis.get_major_locator().set_params(integer=True)
            ax.tick_params(labelsize=11)
            
    plot_on_ax(axs[0, 0], csvs_2d, "dec")
    axs[0, 0].set_title(r"$\mathbf{2D}$" + "\n\nSpectral Radii", fontsize=15)
    axs[0, 0].set_ylim(0.0, 1.02)
    axs[0, 0].set_ylabel(r"$|\lambda|$", fontsize=13)
    
    plot_on_ax(axs[0, 1], csvs_3d, "dec")
    axs[0, 1].set_title(r"$\mathbf{3D}$" + "\n\nSpectral Radii", fontsize=15)
    axs[0, 1].set_ylim(0.0, 1.02)
    
    plot_on_ax(axs[1, 0], csvs_2d, "gmres")
    axs[1, 0].set_title("GMRES Iterations", fontsize=14)
    axs[1, 0].set_ylabel("Steps", fontsize=13)
    axs[1, 0].set_xlabel("Refinement Level", fontsize=13)
    
    plot_on_ax(axs[1, 1], csvs_3d, "gmres")
    axs[1, 1].set_title("GMRES Iterations", fontsize=14)
    axs[1, 1].set_xlabel("Refinement Level", fontsize=13)
    
    def get_unique_handles_labels(ax_list):
        unique = {}
        for ax in ax_list:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                base_name = label.split(' ($n_\\mathrm')[0]
                if base_name not in unique or ("($n_\\mathrm" in label and "($n_\\mathrm" not in unique[base_name][1]):
                    unique[base_name] = (handle, label)
        return [v[0] for v in unique.values()], [v[1] for v in unique.values()]
        
    handles_2d, labels_2d = get_unique_handles_labels([axs[0, 0], axs[1, 0]])
    handles_3d, labels_3d = get_unique_handles_labels([axs[0, 1], axs[1, 1]])
    
    fig.subplots_adjust(hspace=0.25, wspace=0.18, bottom=0.22)
    
    if labels_2d:
        add_sorted_legend(axs[1, 0], handles_2d, labels_2d, loc='upper center', 
                          bbox_to_anchor=(0.5, -0.22), fontsize=14, ncol=2, 
                          frameon=True, borderpad=0.6, handlelength=2.0, handletextpad=0.4, columnspacing=0.8)
    if labels_3d:
        add_sorted_legend(axs[1, 1], handles_3d, labels_3d, loc='upper center', 
                          bbox_to_anchor=(0.5, -0.22), fontsize=14, ncol=2, 
                          frameon=True, borderpad=0.6, handlelength=2.0, handletextpad=0.4, columnspacing=0.8)
        
    fig.savefig(output_dir / "summary_2x2.pdf", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

# ==========================================
#              MAIN EXECUTION
# ==========================================

def process_recursive(root_dir, output_root):
    root, out_root = Path(root_dir), Path(output_root)
    if not root.exists(): return

    msh_map = {f.stem: f for f in root.rglob("*.msh")}
    dirs_map, prob_map, unique_stems = {}, {}, set()

    for f in root.rglob("*"):
        if "out" not in f.parts and "meshes" not in f.parts: continue
        if f.is_file() and f.suffix in ['.csv', '.msh']:
            unique_stems.add(f.stem)
            if f.parent not in dirs_map: dirs_map[f.parent] = {'csv': [], 'msh': []}
            dirs_map[f.parent]['csv' if f.suffix == '.csv' else 'msh'].append(f)
            if f.suffix == '.csv' and f.parent.name in ['2d', '3d']:
                prob_dir = f.parent.parent
                if prob_dir not in prob_map: prob_map[prob_dir] = {'2d': [], '3d': []}
                prob_map[prob_dir][f.parent.name].append(f)

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    style_dict = {stem: (colors[i % 20], ['o','s','^','D'][i % 4], ['-','--','-.'][i % 3]) for i, stem in enumerate(sorted(list(unique_stems)))}

    tasks = []
    for directory, file_types in dirs_map.items():
        target_dir = out_root / (directory.relative_to(root) if directory != root else ".")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for msh_file in file_types['msh']: 
            tasks.append((plot_mesh_standalone, (msh_file, target_dir)))
            
        if file_types['csv']:
            tasks.append((plot_dec, (directory, file_types['csv'], target_dir, style_dict)))
            tasks.append((plot_gmres, (directory, file_types['csv'], target_dir, style_dict)))
            
            for csv_file in file_types['csv']:
                matched_msh = None
                for m_stem in sorted(msh_map.keys(), key=len, reverse=True):
                    if csv_file.stem.startswith(m_stem):
                        matched_msh = msh_map[m_stem]
                        break
                
                if matched_msh:
                    tasks.append((plot_single_mesh_1x2, (csv_file, matched_msh, target_dir, style_dict)))

    for prob_dir, dims in prob_map.items():
        tasks.append((plot_summary_2x2, (prob_dir, dims['2d'], dims['3d'], out_root / prob_dir.relative_to(root), style_dict)))

    if tasks:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(func, *args) for func, args in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Rendering Plots & 3D Meshes"): pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs='?', default=".")
    parser.add_argument("-o", "--output", default="plots_combined")
    args = parser.parse_args()
    process_recursive(args.directory, args.output)
