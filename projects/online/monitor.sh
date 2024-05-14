#!/bin/bash

# trained model weights
export AMPLFI_WEIGHTS=
export AFRAME_WEIGHTS=
# location where low latency data
# is streamed, typically /dev/shm/kakfka
export ONLINE_DATADIR=/dev/shm/kafka/
# where results will be writen
export AFRAME_ONLINE_OUTDIR=

AFRAME_CONTAINER_ROOT=~/aframe/images/
config=./config.yaml
log_dir=
mkdir -p $log_dir

export CUDA_VISIBLE_DEVICES=2
crash_count=0
until apptainer run --nv --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" $AFRAME_CONTAINER_ROOT/online.sif poetry run python online/cli.py --config $config 2>> $log_dir/monitoring.log; do
    ((crash_count++))
    echo "Online deployment crashed on $(date) with error code $?, crash count = $crash_count" >> $log_dir/monitoring.log
    sleep 1
done
