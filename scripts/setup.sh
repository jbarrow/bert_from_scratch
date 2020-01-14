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

# BERT parameters
masked_lm_prob=0.15
max_seq_length=128
max_predictions_per_seq=20

# Create the model directory
mkdir $directory

# Install the software requirements (python and BERT)
pip install --user -r structurebert/requirements.txt
git clone https://github.com/google-research/bert

# Download the training and validation data
#gsutil cp gs://contract-bert/processed/contracts_train.txt ./
gsutil cp gs://contract-bert/processed/contracts_eval.txt ./


# Generate the blank BERT model.
python3 structurebert/structurebert/setup.py \
		--model-directory $directory \
		--dataset $validation_dataset \
		--vocab-file vocab.txt \
    --vocab-size 4096

mkdir ./shards
split -a 4 -l 256000 -d processed.txt ./shards/shard_

python3 bert/create_pretraining_data.py \
  --input_file=processed.txt \
  --output_file=train.tfrecord \
  --vocab_file=vocab.txt \
  --do_lower_case=True \
  --do_whole_word_mask=True \
  --max_predictions_per_seq=$max_predictions_per_seq \
  --max_seq_length=$max_seq_length \
  --masked_lm_prob=$masked_lm_prob \
  --random_seed=34 \
  --dupe_factor=5
  
