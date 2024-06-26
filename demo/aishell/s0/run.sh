#!/bin/bash

. ./path.sh || exit 1;
echo $PWD
export CUDA_VISIBLE_DEVICES="0,1,2"
export OMP_NUM_THREADS='1'
export MKL_NUM_THREADS='1'

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi

data_dir="E:\data\aishell-200\data_aishell\data_aishell_test"

stop_stage=0
start_stage=0


if [ $stop_stage -ge 0 ] and [ $start_stage -le 0 ]; then
    python fqdd/bin/asr/train.py
fi

if [ $stop_stage -ge 1 ] and [ $start_stage -le 1 ]; then
    python fqdd/bin/asr/train.py
fi

if [ $stop_stage -ge 2 ] and [ $start_stage -le 2 ]; then
    python fqdd/bin/asr/train.py
fi

if [ $stop_stage -ge 3 ] and [ $start_stage -le 3 ]; then
    python fqdd/bin/asr/train.py
fi

if [ $stop_stage -ge 4 ] and [ $start_stage -le 4 ]; then
    echo "pass"
fi

if [ $stop_stage -ge 5 ] and [ $start_stage -le 5 ]; then
    python fqdd/bin/asr/train.py
fi

if [ $stop_stage -ge 6 ] and [ $start_stage -le 6 ]; then
    echo "pass"
fi

if [ $stop_stage -ge 7 ] and [ $start_stage -le 7 ]; then
    echo "pass"
fi

if [ $stop_stage -ge 8 ] and [ $start_stage -le 8 ]; then
    echo "pass"
fi

if [ $stop_stage -ge 9 ] and [ $start_stage -le 9 ]; then
    echo "pass"
fi
# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=-1 --node-rank=0 fqdd/bin/asr/train.py

