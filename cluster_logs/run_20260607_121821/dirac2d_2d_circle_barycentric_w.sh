#!/bin/bash
#SBATCH --job-name=dirac2d_2d_circle_barycentric_w
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096M
#SBATCH --time=00:45:00
#SBATCH --output=cluster_logs/run_20260607_121821/dirac2d_2d_circle_barycentric_w.out
#SBATCH --error=cluster_logs/run_20260607_121821/dirac2d_2d_circle_barycentric_w.err

module load stack gcc openmpi eigen boost cmake python

/cluster/home/rdabetic/decmg/release/targets/cvg --config /cluster/home/rdabetic/decmg/configs/dirac2d_2d_circle_barycentric_w.json
