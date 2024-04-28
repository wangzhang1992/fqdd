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
python fqdd/bin/asr/train.py


# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=-1 --node-rank=0 fqdd/bin/asr/train.py

