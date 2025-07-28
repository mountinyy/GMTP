#!/bin/sh

# Manual Config -----------------------------------------------
# General settings
generator="llama2"
retrieval_only=True
latency_check=True
debug=False
api_key="YOUR_API_KEY"

# Attack settings
do_attack=True
attack="poisonedrag" # poisonedrag phantom advdecoding
dataset="nq" # nq hotpotqa msmarco
retriever="dpr" # dpr contriever

# Defense settings
defense_method=gmtp #  gmtp perplexity l2  
reranker="bert" # bert roberta  
N=10 # N
M=5 # M
remove_threshold=-1
remove_threshold_path="data/mask_probs/${attack}/${dataset}_${retriever}_${reranker}_${N}_min${M}.json"
remove_lambda=0.1
use_random_doc=True

# RAG settings
retrieve_k=20 # Retrieved passgaes.
rerank_k=10 # Actual used passages.
# --------------------------------------------------------------


if [ $attack == "advdecoding" ]; then
  attack_method="trigger_append" # hotflip trigger_append augmented bert
else
  attack_method="hotflip"
fi


###############
# Merge Index #
###############

# For no defense, no rerank. Directly retrieve k-documents instead of 2k-documents and reraning  top-k.

# Query

if [ $attack == "phantom" ]; then
  query="data/poisoned_documents/${attack}/${attack_method}/${retriever}/${generator}/${dataset}/queries.tsv"
else
  if [ $defense_method == "paraphrased" ]; then
    query="data/poisoned_documents/${attack}/${attack_method}/${retriever}/${dataset}_paraphrased/queries.tsv"
  else
    query="data/poisoned_documents/${attack}/${attack_method}/${retriever}/${dataset}/queries.tsv"
  fi
fi

if [ $do_attack == True ]; then
  if [ $attack == "phantom" ]; then
    index="data/faiss/${dataset}/embeddings/${retriever}/merged/${attack}_${attack_method}_${generator}"
    output_dir="results/retrieval/${dataset}/${attack}/${attack_method}/${retriever}-${generator}-${retrieve_k}.json"
  else
    if [ $defense_method == "paraphrased" ]; then
      index="data/faiss/${dataset}/embeddings/${retriever}/merged/${attack}_${attack_method}_paraphrased"
      output_dir="results/retrieval/${dataset}/${attack}/${attack_method}_paraphrased/${retriever}-${retrieve_k}.json"
    else
      index="data/faiss/${dataset}/embeddings/${retriever}/merged/${attack}_${attack_method}"
      output_dir="results/retrieval/${dataset}/${attack}/${attack_method}/${retriever}-${retrieve_k}.json"
    fi
  fi
  mkdir -v -p "results/retrieval/${dataset}/${attack}/${attack_method}"
else
  index="data/faiss/${dataset}/embeddings/${retriever}/clean"
  output_dir="results/retrieval/${dataset}/clean_$attack_method/${retriever}-${retrieve_k}.json"
  mkdir -v -p "results/retrieval/${dataset}/clean"
fi

clean_index_dir="data/faiss/$dataset/embeddings/$retriever/clean"

if [ $attack == "phantom" ]; then
    atk_index_dir="data/faiss/$dataset/embeddings/$retriever/${attack}/${attack_method}/${generator}"
else
  if [ $defense_method == "paraphrased" ]; then
    atk_index_dir="data/faiss/$dataset/embeddings/$retriever/${attack}/${attack_method}_paraphrased"
  else
    atk_index_dir="data/faiss/$dataset/embeddings/$retriever/${attack}/${attack_method}"
  fi
fi


if [ $do_attack == True ] && [ ! -f "${index}/index" ]; then
  echo "Create and merge index in $index"
  echo "Clean index: $clean_index_dir"
  echo "Attack index: $atk_index_dir"

  python merge_index.py \
      --index_dir $clean_index_dir \
      --atk_index_dir $atk_index_dir \
      --full_index_dir $index

fi

################
# Faiss Search #
################
if [ -f $output_dir ]; then
  echo "Skip search, $output_dir exists"
else
echo "Search in $output_dir"
  if [ $retriever == "contriever" ]; then
    encoder=facebook/contriever-msmarco
  elif [ $retriever == "dpr" ]; then
    encoder=facebook/dpr-question_encoder-single-nq-base
  fi
echo "Index: $index"
echo "Query: $query"
echo "Encoder: $encoder"
  CUDA_VISIBLE_DEVICES="0"  python src/search.py  \
    --index $index \
    --topics $query \
    --encoder $encoder \
    --output $output_dir \
    --max-length 512 \
    --batch-size 64 --threads 16 --hits $retrieve_k --device cuda

fi
#############
# Evaluation #
#############
echo "Dataset: $dataset | Do_attack : $do_attack | Attack: $attack | Attack Method: $attack_method | Defense Method: $defense_method | "
echo "Retriever: $retriever | Generator: $generator"
python run_main.py \
    ++api_key=$api_key \
    rag.retrieval_result_path=$output_dir \
    attack=$attack \
    attack.attack_method=$attack_method \
    dataset.name=$dataset \
    model.retriever=$retriever \
    model.generator=$generator \
    rag.top_k=$rerank_k \
    common.do_attack=$do_attack \
    defense.method=$defense_method \
    common.debug=$debug \
    common.N=$N \
    common.remove_threshold=$remove_threshold \
    common.remove_threshold_path=$remove_threshold_path \
    model.reranker=$reranker \
    common.remove_lambda=$remove_lambda \
    common.M=$M \
    common.latency_check=$latency_check \
    common.retrieval_only=$retrieval_only \
    common.remove_threshold_random_doc=$use_random_doc \