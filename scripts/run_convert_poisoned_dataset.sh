#!/bin/sh

# Manual Config -----------------------------------------------
attack="phantom" # poisonedrag phantom  advdecoding
total=200
# ----------------------------------------------------------

if [ $attack == "advdecoding" ]; then
    attack_method="trigger_append"
else
    attack_method="hotflip"
fi

generator="llama2" # llama2 mistral7b

datasets=("nq" "hotpotqa" "msmarco")
retrievers=("contriever" "dpr")
for retriever in "${retrievers[@]}"; do
    for dataset in "${datasets[@]}"; do
        root_dir="data/poisoned_documents/${attack}/${attack_method}/${retriever}"
        dataset_dir="${root_dir}/${dataset}-${total}.json"
        output_dir="${root_dir}/${dataset}"
        if [ $attack == "phantom" ]; then
            root_dir="data/poisoned_documents/${attack}/${attack_method}/${retriever}/${generator}"
            dataset_dir="${root_dir}/${dataset}-${total}.json"
            output_dir="${root_dir}/${dataset}"
        fi

        python convert_poisoned_dataset_to_jsonl.py \
            --dataset=$dataset_dir \
            --output_dir=$output_dir \

    done
done