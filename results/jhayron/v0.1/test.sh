#!/bin/tcsh                                                                         
### bash users replace /tcsh with /bash -l
#PBS -N test
#PBS -A ACGD0007
#PBS -l walltime=1:00:00
#PBS -q main
#PBS -j oe
#PBS -k eod
#PBS -m abe
#PBS -l select=2:ncpus=36:mpiprocs=36
 
### Set TMPDIR as recommended
setenv TMPDIR /glade/scratch/$USER/temp
### bash users: export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
 
echo "Hello World!"
### Run the executable
#mpiexec_mpt ./executable_name.exe
