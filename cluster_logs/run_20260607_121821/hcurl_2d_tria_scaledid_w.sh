#!/bin/bash
#SBATCH --job-name=hcurl_2d_tria_scaledid_w
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096M
#SBATCH --time=00:45:00
#SBATCH --output=cluster_logs/run_20260607_121821/hcurl_2d_tria_scaledid_w.out
#SBATCH --error=cluster_logs/run_20260607_121821/hcurl_2d_tria_scaledid_w.err

module load stack gcc openmpi eigen boost cmake python

/cluster/home/rdabetic/decmg/release/targets/cvg --config /cluster/home/rdabetic/decmg/configs/hcurl_2d_tria_scaledid_w.json
