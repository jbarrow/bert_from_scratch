#!/bin/bash

# Script to install the necessary requirements, download the data, and setup an
# empty BERT model with a vocab based on the dataset.

# ARGUMENTS
# $1 - bucket name (we'll use this both locally and on GCE)
# $2 - tpu name

bucket_name=$1
tpu_name=$2

train_dataset=contracts_train.txt
validation_dataset=contracts_eval.txt

# BERT parameters
masked_lm_prob=0.15
max_seq_length=128
max_predictions_per_seq=20
vocab_size=32000

# Create the model directory

data_directory=data
proc_directory=$bucket_name/processed
bert_directory=$bucket_name/tf1.0
tf_directory=$bucket_name/tfrecords

# mkdir $bucket_name $data_directory $proc_directory $bert_directory $tf_directory
#
# # Install the software requirements (python and BERT)
# pip3 install --user -r bert_from_scratch/requirements.txt
# git clone https://github.com/google-research/bert
#
# # Download the training and validation data
# gsutil cp gs://contract-bert/processed/$train_dataset $data_directory/
# gsutil cp gs://contract-bert/processed/$validation_dataset $data_directory/
#
#
# # Generate the blank BERT model.
# python3 bert_from_scratch/structurebert/initialize_original_tf_bert.py \
# 		--model-directory $bert_directory \
# 		--train-dataset $data_directory/$train_dataset \
#     --validation-dataset $data_directory/$validation_dataset \
# 		--vocab-file vocab.txt \
#     --vocab-size $vocab_size \
#     --processed-output-directory $proc_directory \
#     --do-lower-case

mkdir $proc_directory/shards
split -a 4 -l 256000 -d $proc_directory/train.txt $proc_directory/shards/shard_

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

python3 bert/run_pretraining.py \
  --input_file=gs://$tf_directory/train.tfrecord \
  --output_dir=gs://$tf_directory/pretraining_output \
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
