#!/bin/bash
#SBATCH --job-name=dirac3d_3d_corner_barycentric_w
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1024M
#SBATCH --time=00:05:00
#SBATCH --output=cluster_logs/run_20260607_113846/dirac3d_3d_corner_barycentric_w.out
#SBATCH --error=cluster_logs/run_20260607_113846/dirac3d_3d_corner_barycentric_w.err

module load stack gcc openmpi boost
/cluster/home/rdabetic/decmg/release/targets/cvg --config /cluster/home/rdabetic/decmg/configs/dirac3d_3d_corner_barycentric_w.json
