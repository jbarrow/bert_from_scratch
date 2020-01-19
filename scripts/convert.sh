#!/bin/bash
# ARGUMENTS
# $1 - bucket name

bucket_name=$1

mkdir $bucket_name $bucket_name/tf1.0 $bucket_name/pytorch

gsutil -m cp gs://$bucket_name/tf1.0 $bucket_name/tf1.0

cp $bucket_name/tf1.0/vocab.txt $bucket_name/pytorch/vocab.txt
cp $bucket_name/tf1.0/bert_config.json $bucket_name/pytorch/config.json

python3 -m transformers.convert_bert_original_tf_checkpoint_to_pytorch \
  --tf_checkpoint_path $bucket_name/tf1.0/model.ckpt \
  --bert_config_file $bucket_name/bert_config.json \
  --pytorch_dump_path $bucket_name/pytorch/pytorch_model.bin
