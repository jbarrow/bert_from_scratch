#!/bin/bash

# ARGUMENTS
# $1 - bucket name (we'll use this both locally and on GCE)
# $2 - text file to train from
# $3 - bert directory

bucket_name=$1
train_dataset=$2
bert_directory=$3

# BERT parameters
masked_lm_prob=0.15
max_seq_length=512
max_predictions_per_seq=80
vocab_size=32000

# Create the model directory

data_directory=data
proc_directory=$bucket_name/processed
tf_directory=$bucket_name/tfrecords

mkdir -p $bucket_name \
        $data_directory \
        $proc_directory \
        $tf_directory \
        $proc_directory/shards

split -a 4 -l 256000 -d $train_dataset $proc_directory/shards/shard_

# Generate the training data
ls $proc_directory/shards/ | xargs -n 1 -P 24 -I{} python3 bert/create_pretraining_data.py \
  --input_file=$proc_directory/shards/{} \
  --output_file=$tf_directory/{}.tfrecord \
  --vocab_file=$bert_directory/vocab.txt \
  --do_lower_case=True \
  --do_whole_word_mask=True \
  --max_predictions_per_seq=$max_predictions_per_seq \
  --max_seq_length=$max_seq_length \
  --masked_lm_prob=$masked_lm_prob \
  --random_seed=34 \
  --dupe_factor=5

gsutil -m cp -r $bucket_name/* gs://$bucket_name/
