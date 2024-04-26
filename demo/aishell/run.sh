nohup python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node-rank=0 script/asr/train.py > log.txt 2>&1 &
