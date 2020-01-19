#!/bin/bash

# Script to launch BERT training on a TPU.

# ARGUMENTS
# $1 - bucket name (we'll use this both locally and on GCE)
# $2 - tpu name

bucket_name=$1
tpu_name=$2

bert_directory=$bucket_name/tf1.0
tf_directory=$bucket_name/tfrecords

python3 bert/run_pretraining.py \
  --input_file=gs://$tf_directory/*.tfrecord \
  --output_dir=gs://$bert_directory \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$bert_directory/bert_config.json \
  --train_batch_size=128 \
  --eval_batch_size=64 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000000 \
  --save_checkpoint_steps=2500 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=True \
  --tpu_name=$tpu_name
