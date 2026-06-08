#!/bin/bash
#SBATCH --job-name=dirac3d_3d_cube_scaledid_w
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16384M
#SBATCH --time=02:00:00
#SBATCH --output=cluster_logs/run_20260607_121821/dirac3d_3d_cube_scaledid_w.out
#SBATCH --error=cluster_logs/run_20260607_121821/dirac3d_3d_cube_scaledid_w.err

module load stack gcc openmpi eigen boost cmake python

/cluster/home/rdabetic/decmg/release/targets/cvg --config /cluster/home/rdabetic/decmg/configs/dirac3d_3d_cube_scaledid_w.json
