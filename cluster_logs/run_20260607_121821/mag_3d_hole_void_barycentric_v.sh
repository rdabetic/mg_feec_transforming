#!/bin/bash
#SBATCH --job-name=mag_3d_hole_void_barycentric_v
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16384M
#SBATCH --time=02:00:00
#SBATCH --output=cluster_logs/run_20260607_121821/mag_3d_hole_void_barycentric_v.out
#SBATCH --error=cluster_logs/run_20260607_121821/mag_3d_hole_void_barycentric_v.err

module load stack gcc openmpi eigen boost cmake python

/cluster/home/rdabetic/decmg/release/targets/cvg --config /cluster/home/rdabetic/decmg/configs/mag_3d_hole_void_barycentric_v.json
