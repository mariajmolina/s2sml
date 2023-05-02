#!/bin/bash -l
#PBS -N s2sml-kdagon
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB
#PBS -l walltime=2:00:00
#PBS -l gpu_type=a100
#PBS -A ACGD0007
#PBS -q main
#PBS -o /glade/work/kdagon/s2sml/results/out
#PBS -e /glade/work/kdagon/s2sml/results/err
source ~/.bashrc; module load conda; conda activate s2sml-env
echo-run hyperKDGust.yml unetKDGust.yml -n $PBS_JOBID
