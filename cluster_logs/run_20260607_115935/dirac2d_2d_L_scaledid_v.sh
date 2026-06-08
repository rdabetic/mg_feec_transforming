#!/bin/bash
#SBATCH --job-name=dirac2d_2d_L_scaledid_v
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=256M
#SBATCH --time=00:05:00
#SBATCH --output=cluster_logs/run_20260607_115935/dirac2d_2d_L_scaledid_v.out
#SBATCH --error=cluster_logs/run_20260607_115935/dirac2d_2d_L_scaledid_v.err

module load stack gcc openmpi boost
/cluster/home/rdabetic/decmg/release/targets/cvg --config /cluster/home/rdabetic/decmg/configs/dirac2d_2d_L_scaledid_v.json
