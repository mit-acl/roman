#!/usr/bin/env bash
# Usage: ./run.bash [data_dir] [results_dir] [params_dir] [gpu_arg]
# Mounts data_dir to /root/data
# Mounts results_dir to /root/results
# Mounts params_dir to /root/params
# gpu_arg examples: " " (no gpu), "--gpus all"

export DATA_DIR=${1:-$ROMAN_DEMO_DATA}
export RESULTS_DIR=${2:-"./"}
export PARAMS_DIR=${3:-"./params"}
export GPU_ARG=${4:-"--gpus all"}

VOLUMES="-v $DATA_DIR:/root/data -v $RESULTS_DIR:/root/results -v $PARAMS_DIR:/root/params"
VARS="-e ROMAN_DEMO_DATA=/root/data -e ROMAN_WEIGHTS=/root/roman/weights"

docker run -it --rm --net=host \
    $VOLUMES $VARS $GPU_ARG roman