#!/bin/bash -l
#PBS -N s2sml-jhayron
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB
#PBS -l walltime=12:00:00
#PBS -l gpu_type=a100
#PBS -A ACGD0007
#PBS -q main
#PBS -o /glade/u/home/jhayron/s2sml/results/jhayron/v0.1/out
#PBS -e /glade/u/home/jhayron/s2sml/results/jhayron/v0.1/out
echo 'hello world'


