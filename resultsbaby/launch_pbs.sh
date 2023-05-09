#!/bin/bash -l
#PBS -N s2sml-kdagon
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB
#PBS -l walltime=6:00:00
#PBS -l gpu_type=v100
#PBS -A ACGD0007
#PBS -q casper
#PBS -o /glade/work/kdagon/s2sml/resultsbaby/out
#PBS -e /glade/work/kdagon/s2sml/resultsbaby/err
source ~/.bashrc; module unload cuda; conda activate s2sml-env
echo-run hyperbaby2.yml unetbaby2.yml -n $PBS_JOBID
