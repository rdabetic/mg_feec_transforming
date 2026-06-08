#!/bin/bash
#SBATCH --job-name=hcurl_3d_hole_void_scaledid_v
#SBATCH --constraint=EPYC_9654
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1024M
#SBATCH --time=00:05:00
#SBATCH --output=cluster_logs/run_20260607_115935/hcurl_3d_hole_void_scaledid_v.out
#SBATCH --error=cluster_logs/run_20260607_115935/hcurl_3d_hole_void_scaledid_v.err

module load stack gcc openmpi boost
/cluster/home/rdabetic/decmg/release/targets/cvg --config /cluster/home/rdabetic/decmg/configs/hcurl_3d_hole_void_scaledid_v.json
