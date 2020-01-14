#!/bin/bash

bucket_name=contractbert-128-uncased

mkdir $bucket_name $bucket_name/tf1.0 $bucket_name/pytorch

gsutil -m cp gs://$bucket_name/tf1.0 $bucket_name/tf1.0

python3 -m transformers.convert_bert_original_tf_checkpoint_to_pytorch \
  --tf_checkpoint_path $bucket_name/model.ckpt-0 \
  --bert_config_file $bucket_name/bert_config.json \
  --pytorch_dump_path $bucket_name/pytorch/$bucket_name
