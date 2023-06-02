#!/bin/bash -l
#PBS -N s2sml-kdagon
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB:host=gu0017
#PBS -l walltime=4:00:00
#PBS -l gpu_type=a100
#PBS -A ACGD0007
#PBS -q main
#PBS -o /glade/work/kdagon/s2sml/testGustInternet_results/out
#PBS -e /glade/work/kdagon/s2sml/testGustInternet_results/err
source ~/.bashrc; conda activate s2sml-env
echo-run hyperKDGustIntTest.yml unetKDGustIntTest.yml -n $PBS_JOBID
