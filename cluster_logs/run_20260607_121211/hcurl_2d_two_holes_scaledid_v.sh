#!/bin/bash
#SBATCH --job-name=hcurl_2d_two_holes_scaledid_v
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=256M
#SBATCH --time=00:05:00
#SBATCH --output=cluster_logs/run_20260607_121211/hcurl_2d_two_holes_scaledid_v.out
#SBATCH --error=cluster_logs/run_20260607_121211/hcurl_2d_two_holes_scaledid_v.err

module load stack gcc openmpi eigen boost cmake python

/cluster/home/rdabetic/decmg/release/targets/cvg --config /cluster/home/rdabetic/decmg/configs/hcurl_2d_two_holes_scaledid_v.json
