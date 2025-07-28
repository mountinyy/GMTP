#!/bin/sh

augmented=False
datasets=("nq" "hotpotqa" "msmarco")

for dataset in "${datasets[@]}"; do
    if [ $augmented == True ]; then
        output_dir="data/beir/${dataset}_augmented"
        python convert_dataset_to_jsonl.py \
            --dataset=$dataset \
            --output_dir=$output_dir \
            --augmented
    else
        output_dir="data/beir/${dataset}"
        python convert_dataset_to_jsonl.py \
            --dataset=$dataset \
            --output_dir=$output_dir \
            
    fi

done