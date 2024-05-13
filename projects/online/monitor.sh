#!/bin/bash

AFRAME_CONTAINER_ROOT=/home/william.benoit/aframe/images/
config=./config.yaml
log_dir=/home/william.benoit/online_monitoring
mkdir -p $log_dir

export CUDA_VISIBLE_DEVICES=2
crash_count=0
until apptainer run $AFRAME_CONTAINER_ROOT/online.sif poetry run python online/cli.py --config $config 2>> $log_dir/monitoring.log; do
    ((crash_count++))
    echo "Online deployment crashed on $(date) with error code $?, crash count = $crash_count" >> $log_dir/monitoring.log
    sleep 1
done
