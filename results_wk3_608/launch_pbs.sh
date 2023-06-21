#!/bin/bash -l
#PBS -N s2smlwk3
#PBS -l select=1:ncpus=4:ngpus=4:mem=30GB
#PBS -l walltime=12:00:00
#PBS -A ACGD0007
#PBS -q main
#PBS -o /glade/work/kdagon/s2sml/results_wk3_608/out
#PBS -e /glade/work/kdagon/s2sml/results_wk3_608/err
module load conda/latest; conda activate s2sml-env
echo-run hyper_multi_wk3_SH_608.yml unet_multi_wk3_SH_608.yml -n $PBS_JOBID
