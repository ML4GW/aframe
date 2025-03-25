#!/bin/bash
# trained model weights
export AMPLFI_HL_WEIGHTS=
export AMPLFI_HLV_WEIGHTS=
export AFRAME_WEIGHTS=

# file containing timeslide events detected
# by a model with the AFRAME_WEIGHTS above
export ONLINE_BACKGROUND_FILE=
export ONLINE_FOREGROUND_FILE=
export ONLINE_REJECTED_FILE=

# location where low latency data
# is streamed, typically /dev/shm/kakfka
export ONLINE_DATADIR=/dev/shm/kafka/

# where results and deployment logs will be writen
export AFRAME_ONLINE_OUTDIR=

# where aframe containers are stored
export AFRAME_CONTAINER_ROOT=

config=/path/to/config.yaml

export CUDA_VISIBLE_DEVICES=
crash_count=0
until apptainer run --nv --bind $XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --env AMPLFI_HL_WEIGHTS=$AMPLFI_HL_WEIGHTS --env AMPLFI_HLV_WEIGHTS=$AMPLFI_HLV_WEIGHTS--env AFRAME_ONLINE_OUTDIR=$AFRAME_ONLINE_OUTDIR --env ONLINE_DATADIR=$ONLINE_DATADIR --env AFRAME_WEIGHTS=$AFRAME_WEIGHTS --env AMPLFI_WEIGHTS=$AMPLFI_WEIGHTS $AFRAME_CONTAINER_ROOT/online.sif /opt/env/bin/online --config $config 2>> monitoring.log; do
    ((crash_count++))
    echo "Online deployment crashed on $(date) with error code $? from $(hostname),
    crash count = $crash_count" >> monitoring.log
    sleep 1
done
