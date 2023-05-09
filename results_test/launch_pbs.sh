#!/bin/bash -l
#PBS -N s2sml-kdagon
#PBS -l select=1:ncpus=1:ngpus=1:mem=32GB
#PBS -l walltime=1:00:00
#PBS -l gpu_type=v100
#PBS -A acgd0007
#PBS -q casper
#PBS -o /glade/work/kdagon/s2sml/results/out
#PBS -e /glade/work/kdagon/s2sml/results/err
source ~/.bashrc; module unload cuda; conda activate s2sml-env
echo-run config/hyperKD.yml config/unetKD.yml -n $PBS_JOBID
