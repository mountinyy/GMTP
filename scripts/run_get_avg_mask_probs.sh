#!/bin/sh
# Manual Config -----------------------------------------------
attack="phantom" # phantom poisonedrag advdecoding
dataset="nq" # nq hotpotqa msmarco 
retriever="dpr" # dpr contriever
reranker="bert" # bert roberta
N=10 # 1 3 5
M=5

total_samples=1000
use_random_doc=False
include_poison=True # False True
# ------------------------------------------------------------

seed=42
poison_doc_path="data/poisoned_documents/${attack}/hotflip/${retriever}/${dataset}-200.json"

if [ $attack == "phantom" ]; then
    poison_doc_path="data/poisoned_documents/${attack}/hotflip/${retriever}/gemma2/${dataset}-200.json"
fi

if [ $use_random_doc == True ]; then
    python get_avg_mask_probs.py \
        --dataset $dataset \
        --retriever $retriever \
        --poison_doc_path $poison_doc_path \
        --N $N \
        --seed $seed \
        --total_samples $total_samples \
        --attack $attack \
        --reranker $reranker \
        --M $M \
        --include_poison \
        --random_doc
else
    python get_avg_mask_probs.py \
        --dataset $dataset \
        --retriever $retriever \
        --poison_doc_path $poison_doc_path \
        --N $N \
        --seed $seed \
        --total_samples $total_samples \
        --attack $attack \
        --reranker $reranker \
        --M $M \
        --include_poison \   
fi