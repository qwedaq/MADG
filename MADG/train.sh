#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="enter path"
PROJ_NAME="DG_OH"
LOG_FILE="${PROJ_ROOT}/log_OH_R/${PROJ_NAME}-`date +'%Y-%m-%d-%H-%M-%S'`.log"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python3 ${PROJ_ROOT}/train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset Office-Home \
    >> ${LOG_FILE}  2>&1
