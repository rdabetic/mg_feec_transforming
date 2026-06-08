#!/bin/bash
#SBATCH --job-name=cvg_build
#SBATCH --ntasks=1 --cpus-per-task=16 --mem-per-cpu=1G --time=00:15:00 --constraint=EPYC_9654
#SBATCH --output=cluster_logs/run_20260607_113846/build.out
#SBATCH --error=cluster_logs/run_20260607_113846/build.err
module load stack gcc openmpi eigen boost cmake python
cmake -B release -S . -DCMAKE_BUILD_TYPE=Release && cmake --build release -j 16
