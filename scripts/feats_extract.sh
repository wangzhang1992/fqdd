#!/bin/bash
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh

nj=16
datapath=$1
stage=2
feat_type=mfcc_pitch

  if [ $stage -le 0 ]; then
        rm -rf data && mkdir data
        for x in train test dev; do
           python3 scripts/prepare_data.py $datapath/${x}/  data/${x}_raw
           utils/utt2spk_to_spk2utt.pl data/${x}_raw/utt2spk > data/${x}_raw/spk2utt
           utils/fix_data_dir.sh data/${x}_raw
        done
        mv data/test_raw data/test && mv data/dev_raw data/dev
        utils/data/perturb_data_dir_speed_3way.sh data/train_raw data/train
        utils/data/perturb_data_dir_volume.sh data/train
  fi

  if [ $stage -le 1 ]; then
        #produce MFCC features
        echo "produce MFCC features"
        rm -rf data/$feat_type && mkdir -p data/$feat_type &&  cp -r data/train data/$feat_type/train && cp -r data/test data/$feat_type/test &&  cp -r data/dev data/$feat_type/dev || exit 1;
        #rm -rf data/mfcc_hires && mkdir -p data/mfcc_hires &&  cp -r data/{train,dev} data/mfcc_hires || exit 1;

     for x in train; do
        echo "make normal  mfcc"
        steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf --nj $nj data/$feat_type/$x exp/make_$feat_type $feat_type/$x

        #steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj --cmd "$train_cmd" data/mfcc_hires/$x exp/mfcc_hires/$x mfcc_hires/$x || exit 1;
        #compute cmvn
        steps/compute_cmvn_stats.sh data/$feat_type/$x exp/$feat_type/$x $feat_type/$x || exit 1;
     done

   fi


   if [ $stage -le 2 ]; then

      feature_transform_proto=$feat_type/proto
      splice=0
      train_feats="ark:copy-feats scp:data/$feat_type/train/feats.scp ark:- | apply-cmvn --utt2spk=ark:data/$feat_type/train/utt2spk scp:data/$feat_type/train/cmvn.scp ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |"

      feat_dim=$(feat-to-dim "$train_feats" -)
      echo $feat_dim > "$feat_type/feat_dim"
      echo "<Splice> <InputDim> $feat_dim <OutputDim> $(((2*splice+1)*feat_dim)) <BuildVector> -$splice:$splice </BuildVector>" >$feature_transform_proto

      feature_transform=$feat_type/nnet
      nnet-initialize --binary=false $feature_transform_proto $feature_transform

      nnet-forward --print-args=true $feature_transform "$train_feats" ark:- |\
      compute-cmvn-stats ark:- - | cmvn-to-nnet --std-dev=1.0 - -| nnet-concat --binary=false $feature_transform - "$feat_type/final.feature_transform"

      # copy-feats scp:data/$feat_type/train/feats.scp ark:- | apply-cmvn --utt2spk=ark:data/$feat_type/train/utt2spk scp:data/$feat_type/train/cmvn.scp ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- | nnet-forward final.feature_transform ark:- ark:- |
  fi
