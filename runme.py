#!/usr/bin/env python3
"""
Simple Async SLURM Scheduler.
Submits standard bash scripts and manages cluster headroom.
If any job fails, the entire sweep is aborted and active jobs are cancelled.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

MAX_CLUSTER_RAM_GB = 1024
CPUS_PER_TASK      = 4
NODE_CONSTRAINT    = "EPYC_9654"

def parse_mem(mem_str):
    """Parse memory string like '20G', '512M' into MB integer."""
    m = re.match(r'^(\d+)([GMK])?$', mem_str.strip(), re.IGNORECASE)
    if not m: return 16 * 1024
    val = int(m.group(1))
    unit = (m.group(2) or 'G').upper()
    multipliers = {'G': 1024, 'M': 1, 'K': 1 / 1024}
    return max(1, int(val * multipliers[unit]))

def scan_configs(configs_dir):
    """Find all JSON config files directly inside the flat configuration directory."""
    configs = []
    for p in sorted(Path(configs_dir).glob("*.json")):
        with open(p) as fh:
            cfg = json.load(fh)
        cfg['_path'] = str(p.resolve())
        cfg['_stem'] = p.stem  
        configs.append(cfg)
    return configs

class ClusterResourceManager:
    def __init__(self, max_mem_mb):
        self.max_mem = max_mem_mb
        self.used_mem = 0
        self.cond = asyncio.Condition()

    async def acquire(self, mem):
        async with self.cond:
            await self.cond.wait_for(lambda: self.used_mem + mem <= self.max_mem)
            self.used_mem += mem

    async def release(self, mem):
        async with self.cond:
            self.used_mem -= mem
            self.cond.notify_all()

async def submit_task(task_idx, cfg, log_dir, cvg_exe, rm, active_jobs):
    """Submit a standard bash script, wait for completion, and catch failures immediately."""
    out_csv = cfg.get('output_file')
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    
    mem_mb = parse_mem(cfg.get('slurm_mem', '16G'))
    mem_per_cpu = max(1, mem_mb // CPUS_PER_TASK)
    stem = cfg['_stem']
    
    if mem_mb > rm.max_mem:
        raise ValueError(f"Task '{stem}' requests {mem_mb//1024}G, which exceeds the max cluster limit of {rm.max_mem//1024}G!")

    await rm.acquire(mem_mb)
    
    job_id = None
    proc = None
    
    try:
        script_path = log_dir / f"{stem}.sh"
        out_path = log_dir / f"{stem}.out"
        err_path = log_dir / f"{stem}.err"
        
        script_path.write_text(f"""#!/bin/bash
#SBATCH --job-name={stem}
#SBATCH --constraint={NODE_CONSTRAINT}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem-per-cpu={mem_per_cpu}M
#SBATCH --time={cfg.get('slurm_time', '00:45:00')}
#SBATCH --output={out_path}
#SBATCH --error={err_path}

module load stack gcc openmpi eigen boost cmake python

{cvg_exe} --config {cfg['_path']}
""")
        
        proc = await asyncio.create_subprocess_exec(
            'sbatch', '--wait', str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            line_str = line.decode().strip()
            m = re.search(r'job (\d+)', line_str, re.IGNORECASE)
            if m:
                job_id = m.group(1)
                active_jobs[stem] = job_id
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Submitted ({task_idx:03d}): {stem} (Job {job_id}) | Usage: {rm.used_mem//1024}G / {rm.max_mem//1024}G")
                break
                
        stdout_bytes, stderr_bytes = await proc.communicate()
        
        if proc.returncode != 0:
            err_msg = stderr_bytes.decode().strip()
            err_text = f"❌ FATAL ERROR: Task '{stem}' (Job {job_id}) FAILED!\n"
            err_text += f"   Exit Code: {proc.returncode}\n"
            if err_msg:
                err_text += f"   sbatch Stderr: {err_msg}\n"
            err_text += f"   Check the simulation error log:  {err_path}\n"
            err_text += f"   Check the simulation output log: {out_path}\n"
            raise Exception(err_text)
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed ({task_idx:03d}): {stem} | Freed {mem_mb//1024}G")

    except asyncio.CancelledError:
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
        if job_id:
            print(f"   [Abort] Cancelling active SLURM job {job_id} ({stem})...")
            subprocess.Popen(['scancel', str(job_id)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise
        
    finally:
        active_jobs.pop(stem, None)
        await rm.release(mem_mb)

def main():
    parser = argparse.ArgumentParser(description="Run cvg parameter sweeps via SLURM")
    parser.add_argument("--configs", default="configs", help="Directory containing JSON configs")
    parser.add_argument("--exe", default="./release/targets/cvg", help="Path to cvg executable")
    args = parser.parse_args()

    configs_dir = Path(args.configs)
    cvg_exe = Path(args.exe).resolve()

    if not configs_dir.exists():
        print(f"Error: Configs directory not found: {configs_dir}")
        sys.exit(1)

    out_dir = Path("out")
    if out_dir.exists():
        print(f"Purging existing outputs directory: {out_dir}/")
        shutil.rmtree(out_dir)

    log_dir = Path('cluster_logs') / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    configs = scan_configs(configs_dir)
    if not configs:
        print(f"Error: No JSON config files discovered inside {configs_dir}")
        sys.exit(1)
        
    print(f"Loaded {len(configs)} configs. Logs directory: {log_dir}\n")

    print("Compiling project on target hardware...")
    build_script = log_dir / 'build.sh'
    build_out = log_dir / 'build.out'
    build_err = log_dir / 'build.err'
    
    build_script.write_text(f"""#!/bin/bash
#SBATCH --job-name=cvg_build
#SBATCH --ntasks=1 --cpus-per-task=16 --mem-per-cpu=1G --time=00:15:00 --constraint={NODE_CONSTRAINT}
#SBATCH --output={build_out}
#SBATCH --error={build_err}

module load stack gcc openmpi eigen boost cmake python
cmake -B release -S . -DCMAKE_BUILD_TYPE=Release && cmake --build release -j 16
""")
    res = subprocess.run(['sbatch', '--wait', str(build_script)], capture_output=True, text=True)
    if res.returncode != 0:
        print("\n❌ FATAL ERROR: Compilation Build Job Failed!")
        print(f"   Check error log:  {build_err}")
        print(f"   Check output log: {build_out}")
        sys.exit(1)
        
    print("Build complete. Initializing cluster queue loop...\n")

    async def run_all():
        rm = ClusterResourceManager(MAX_CLUSTER_RAM_GB * 1024)
        active_jobs = {}
        tasks = [asyncio.create_task(submit_task(i, cfg, log_dir, cvg_exe, rm, active_jobs)) for i, cfg in enumerate(configs)]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print("\n" + "═"*70)
            print("🚨 SWEEP ABORTED DUE TO TASK FAILURE 🚨")
            print("═"*70)
            print(e)
            
            print("\nCancelling all remaining tasks and flushing queue...")
            for t in tasks:
                t.cancel()
                
            await asyncio.gather(*tasks, return_exceptions=True)
            
            print("\nSweep safely shut down. Please fix the error and restart.")
            sys.exit(1)

    asyncio.run(run_all())
    print("\n✔ All sweeps completed successfully.")

if __name__ == '__main__':
    main()
