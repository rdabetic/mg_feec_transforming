#!/bin/bash
#SBATCH --job-name=dirac3d_3d_hole_void_rowsum_v
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1024M
#SBATCH --time=00:05:00
#SBATCH --output=cluster_logs/run_20260607_121211/dirac3d_3d_hole_void_rowsum_v.out
#SBATCH --error=cluster_logs/run_20260607_121211/dirac3d_3d_hole_void_rowsum_v.err

module load stack gcc openmpi eigen boost cmake python

/cluster/home/rdabetic/decmg/release/targets/cvg --config /cluster/home/rdabetic/decmg/configs/dirac3d_3d_hole_void_rowsum_v.json
