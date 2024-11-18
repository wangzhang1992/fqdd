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

nodes=1
master_port=8000


start_stage=6
stop_stage=7


data_dir=/data1/data_management/speech_processing/ASR/audio_raw/language_china/zh-CN/lable/near-field/read/aishell/aishell_1_178hr/data_aishell
train_config=conf/conformer_conf.json

train_set="train"
test_sets="test"
dev_sets="dev"
dict=data/dict/lang_char.txt

average_num=5
decode_methods="ctc_greedy_search"

dir=exp/conformer
checkpoint=exp/conformer/pretrain.pt

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "prepare data"
    local/aishell_data_prep.sh $data_dir/wav $data_dir/transcript
fi


if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # remove the space between the text labels for Mandarin dataset
  echo $start_stage
  for x in ${train_set} ${dev_sets}; do
    cp data/${x}/text data/${x}/text.org
    paste -d " " <(cut -f 1 -d" " data/${x}/text.org) \
      <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
      > data/${x}/text
    rm data/${x}/text.org
  done

  tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp data/${train_set}/wav.scp \
    --out_cmvn data/${train_set}/global_cmvn
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Make a dictionary"
  mkdir -p $(dirname $dict)
  echo "<blank> 0" > ${dict}  # 0 is for "blank" in CTC
  echo "<unk> 1"  >> ${dict}  # <unk> must be 1
  echo "<sos/eos> 2" >> $dict
  tools/text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " \
    | tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' | \
    awk '{print $0 " " NR+2}' >> ${dict}
fi

if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare data, prepare required format"
  for x in ${dev_sets} ${test_sets} ${train_set}; do

      tools/make_raw_list.py data/$x/wav.scp data/$x/text \
        data/$x/data.list
  done
fi

if [ $stop_stage -ge 4 ] && [ $start_stage -le 4 ]; then
    python -m torch.distributed.launch --nproc_per_node=$num_gpus --nnodes=$nodes \
	    --master-port=$master_port \
	    fqdd/bin/asr/train.py \
	    --train_config $train_config \
	    --model_dir $dir \
	    ${checkpoint:+--checkpoint $checkpoint} \
	    --train_data data/$train_set/data.list \
	    --dev_data data/$dev_sets/data.list \

fi


if [ $stop_stage -ge 5 ] && [ $start_stage -le 5 ]; then

    # python fqdd/bin/average_model.py --dst_model=exp/conformer/avg_30.pt --src_path=exp/conformer --num=30 --val_best
    dst_model=$dir/avg_${average_num}.pt

    python fqdd/bin/average_model.py \
	--dst_model $dst_model \
	--src_path $dir \
	--num $average_num \
	--val_best
fi


if [ $stop_stage -ge 6 ] && [ $start_stage -le 6 ]; then
        # 测试模型
    python fqdd/bin/asr/recognize.py --configs=${train_config} \
        --checkpoint=$dir/avg_${average_num}.pt \
        --test_data data/test/data.list \
        --gpu "0" \
        --modes $decode_methods \
        --result_dir $dir
fi

if [ $stop_stage -ge 7 ] && [ $start_stage -le 7 ]; then

    for mode in ${decode_modes}; do
        python tools/compute-wer.py --char=1 --v=1 \
            data/test/text $dir/$mode/text > $dir/$mode/wer
    done
fi
