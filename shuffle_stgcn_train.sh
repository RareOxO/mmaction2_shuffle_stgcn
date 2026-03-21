#!/usr/bin/env bash

CONFIG=configs/skeleton/shuffle_stgcn/shuffle_stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py
WORK_DIR=training_space

python tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR