#!/bin/bash

# Script to install the necessary requirements, download the data, and setup an
# empty BERT model with a vocab based on the dataset.

# ARGUMENTS
# $1 - directory
# $2 - train dataset
# $3 - validation dataset

directory=$1
train_dataset=contracts_trian.txt
validation_dataset=contracts_eval.txt

# Create the model directory
mkdir $directory ; cd $directory

# Install the software requirements (python and BERT)
pip3 install -r ../requirements.txt
git clone https://github.com/google-research/bert

# Download the training and validation data
gsutil cp gs://contract-bert/processed/contracts_train.txt ./
gsutil cp gs://contract-bert/processed/contracts_eval.txt ./

# Generate the blank BERT model.
python3 setup.py --model_directory $directory --data $train_dataset

# mkdir ./shards
# split -a 4 -l 256000 -d $PRC_DATA_FPATH ./shards/shard_
#
# ls ./shards/ |
#              "xargs -n 1 -P {} -I{} "
#              "python3 bert/create_pretraining_data.py "
#              "--input_file=./shards/{} "
#              "--output_file={}/{}.tfrecord "
#              "--vocab_file={} "
#              "--do_lower_case={} "
#              "--max_predictions_per_seq={} "
#              "--max_seq_length={} "
#              "--masked_lm_prob={} "
#              "--random_seed=34 "
#              "--dupe_factor=5"
#
# XARGS_CMD = XARGS_CMD.format(PROCESSES, '{}', '{}', PRETRAINING_DIR, '{}',
#                              VOC_FNAME, DO_LOWER_CASE,
#                              MAX_PREDICTIONS, MAX_SEQ_LENGTH, MASKED_LM_PROB)
#
# tf.gfile.MkDir(PRETRAINING_DIR)
