#!/bin/bash
# ARGUMENTS

tpu_name=jdbarrow
bert_base_dir=contractbert-128

python bert/run_pretraining.py \
  --input_file=gs://contract-bert/train.tfrecord \
  --output_dir=gs://contractbert-128/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$bert_base_dir/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=True \
  --tpu_name=$tpu_name
