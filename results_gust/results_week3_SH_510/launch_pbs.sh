#!/bin/bash -l
#PBS -N s2smlwk3SH
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB:host=gu0017
#PBS -l walltime=2:00:00
#PBS -l gpu_type=a100
#PBS -A ACGD0007
#PBS -q main
#PBS -o /glade/work/kdagon/s2sml/results_week3_SH_510/out
#PBS -e /glade/work/kdagon/s2sml/results_week3_SH_510/err
source ~/.bashrc; conda activate s2sml-env
echo-run hyper_week3_SH_510.yml unet_week3_SH_510.yml -n $PBS_JOBID
