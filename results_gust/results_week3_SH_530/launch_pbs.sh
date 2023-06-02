#!/bin/bash -l
#PBS -N s2smlwk3SH
#PBS -l select=1:ncpus=4:ngpus=1:mem=18GB
#PBS -l walltime=2:00:00
#PBS -l gpu_type=a100
#PBS -A ACGD0007
#PBS -q main
#PBS -o /glade/work/kdagon/s2sml/results_week3_SH_530/out
#PBS -e /glade/work/kdagon/s2sml/results_week3_SH_530/err
source ~/.bashrc; conda activate s2sml-env
echo-run hyper_week3_SH_530.yml unet_week3_SH_530.yml -n $PBS_JOBID
