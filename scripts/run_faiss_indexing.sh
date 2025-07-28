#!/bin/sh

# Manual Setup-----------------------------------------------
is_attack=True # Whether the target dataset is clean / attacked
attack="phantom" # poisonedrag phantom  advdecoding

if [ $attack == "advdecoding" ]; then
      attack_method="trigger_append"
else
      attack_method="hotflip"
fi

dataset=nq # nq hotpotqa msamrco
generator=llama2 
encoder_class="contriever" # dpr contriever
# -----------------------------------------------------------

local_dir="./data/beir/${dataset}"
mkdir -v -p $local_dir

if [ $encoder_class == "dpr" ]; then
      encoder=facebook/dpr-ctx_encoder-single-nq-base
elif [ $encoder_class == "contriever" ]; then
      encoder=facebook/contriever-msmarco
fi

output_dir=./data/faiss/${dataset}/embeddings/${encoder_class}

if [ $is_attack == True ]; then
      if [ $attack == "phantom" ]; then
            local_dir="./data/poisoned_documents/${attack}/${attack_method}/${encoder_class}/${generator}/${dataset}"
            output_dir="${output_dir}/${attack}/${attack_method}/${generator}"
      else
            local_dir="./data/poisoned_documents/${attack}/${attack_method}/${encoder_class}/${dataset}"
            output_dir="${output_dir}/${attack}/${attack_method}"
      fi
else
      output_dir="${output_dir}/clean"
fi

echo "Attack: ${attack}, Dataset: ${dataset}, Encoder: ${encoder_class}, Generator: ${generator}"
echo "Output dir: ${output_dir}"
echo "Local dir: ${local_dir}"
mkdir -v -p $output_dir

shard_num=1

CUDA_VISIBLE_DEVICES="0" python -m pyserini.encode \
input --corpus ${local_dir}/corpus.jsonl \
      --docid-field id \
      --fields text  \
      --delimiter "!@#$%^^&*()ABCDE" \
      --shard-id 0 --shard-num $shard_num \
output --embeddings $output_dir \
      --to-faiss \
encoder --encoder $encoder \
      --encoder-class $encoder_class \
      --fields text \
      --pooling mean \
      --batch 128 \
      --max-length 512 \

