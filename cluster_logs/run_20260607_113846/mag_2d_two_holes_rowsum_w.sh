#!/bin/bash
#SBATCH --job-name=mag_2d_two_holes_rowsum_w
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=256M
#SBATCH --time=00:05:00
#SBATCH --output=cluster_logs/run_20260607_113846/mag_2d_two_holes_rowsum_w.out
#SBATCH --error=cluster_logs/run_20260607_113846/mag_2d_two_holes_rowsum_w.err

module load stack gcc openmpi boost
/cluster/home/rdabetic/decmg/release/targets/cvg --config /cluster/home/rdabetic/decmg/configs/mag_2d_two_holes_rowsum_w.json
