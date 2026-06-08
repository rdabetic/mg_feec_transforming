#!/bin/bash
#SBATCH --job-name=hcurl_2d_two_holes_scaledid_w
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=256M
#SBATCH --time=00:05:00
#SBATCH --output=cluster_logs/run_20260607_115935/hcurl_2d_two_holes_scaledid_w.out
#SBATCH --error=cluster_logs/run_20260607_115935/hcurl_2d_two_holes_scaledid_w.err

module load stack gcc openmpi boost
/cluster/home/rdabetic/decmg/release/targets/cvg --config /cluster/home/rdabetic/decmg/configs/hcurl_2d_two_holes_scaledid_w.json
