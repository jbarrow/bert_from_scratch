#!/bin/bash

# Script to install the necessary requirements, download the data, and setup an
# empty BERT model with a vocab based on the dataset.

# ARGUMENTS
# $1 - bucket name (we'll use this both locally and on GCE)
# $2 - tpu name

bucket_name=$1
tpu_name=jdbarrow

train_dataset=contracts_eval.txt
validation_dataset=contracts_eval.txt

# BERT parameters
masked_lm_prob=0.15
max_seq_length=128
max_predictions_per_seq=20
vocab_size=4096

# Create the model directory

data_directory=data
proc_directory=$bucket_name/processed
bert_directory=$bucket_name/tf1.0
tf_directory=$bucket_name/tfrecords

mkdir $bucket_name $data_directory $proc_directory $bert_directory

# Install the software requirements (python and BERT)
pip install --user -r structurebert/requirements.txt
git clone https://github.com/google-research/bert

# Download the training and validation data
gsutil cp gs://contract-bert/processed/$train_dataset $data_directory/
gsutil cp gs://contract-bert/processed/$validation_dataset $data_directory/


# Generate the blank BERT model.
python3 structurebert/structurebert/initialize_original_tf_bert.py \
		--model-directory $bert_directory \
		--train-dataset $validation_dataset \
    --validation-dataset $validation_dataset \
		--vocab-file vocab.txt \
    --vocab-size $vocab_size \
    --processed-output-directory $proc_directory \
    --do-lower-case

# Generate the training data
python3 bert/create_pretraining_data.py \
  --input_file=$proc_directory/train.txt \
  --output_file=$tf_directory/train.tfrecord \
  --vocab_file=$bert_directory/vocab.txt \
  --do_lower_case=True \
  --do_whole_word_mask=True \
  --max_predictions_per_seq=$max_predictions_per_seq \
  --max_seq_length=$max_seq_length \
  --masked_lm_prob=$masked_lm_prob \
  --random_seed=34 \
  --dupe_factor=5

# Generate the validation data
python3 bert/create_pretraining_data.py \
  --input_file=$proc_directory/validatoin.txt \
  --output_file=$tf_directory/validation.tfrecord \
  --vocab_file=$bert_directory/vocab.txt \
  --do_lower_case=True \
  --do_whole_word_mask=True \
  --max_predictions_per_seq=$max_predictions_per_seq \
  --max_seq_length=$max_seq_length \
  --masked_lm_prob=$masked_lm_prob \
  --random_seed=34 \
  --dupe_factor=5


gsutil -m cp -r $bucket_name/* gs://$bucket_name/


python bert/run_pretraining.py \
  --input_file=gs://$tf_directory/train.tfrecord \
  --output_dir=gs://$tf_directory/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$bert_directory/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=True \
  --tpu_name=$tpu_name
