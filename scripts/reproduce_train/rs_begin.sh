#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

# to reproduced the results in our paper, please use:
TRAIN_IMG_SIZE=640  # 640 or 832
data_cfg_path="configs/data/jl1flight_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/jamma/outdoor/mytrain.py"
n_nodes=1
n_gpus_per_node=1
torch_num_workers=4
batch_size=6
pin_memory=true
exp_name="jl1flight"
# ckpt_path="official"  # "official" or ckpt path to use for training

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=200 \
    --flush_logs_every_n_steps=200 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=1 \
    --benchmark=True \
    --max_epochs=60
