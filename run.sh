#!/bin/sh

# NODES <= 32
NODES=10

# SLURM_NTASKS_PER_NODE <= 16
export SLURM_NTASKS_PER_NODE=10

# TIME = minutes <= 48 hours
TIME=$((60*48))


module load hdf5/1.8.16
module load python/2.7.5
module load julia/gcc/0.4.0

salloc -J UTMPlanner -N $NODES --ntasks-per-node=$SLURM_NTASKS_PER_NODE --mem-per-cpu=400 --qos=normal -t $TIME julia simUTMPlannerV1Parallel.jl append


