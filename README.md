# GMTP

Official repo of ACL 2025 Findings paepr: [Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection](https://arxiv.org/abs/2507.18202).

We also provide 200 poisoned test samples, using [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG), [Phantom](https://arxiv.org/abs/2405.20485), and [Adversarial Decoding](https://github.com/collinzrj/adversarial_decoding).


## 🚀 Features
![](./images/figure0-re.png)


## 🛠️ Setup
Download required Beir datasets (NQ, HotpotQA, MS MARCO).
```
python donwload_datasets.py
```
Build and run docker image.
```
docker build -t gmtp .
docker run --rm --gpus all -it -v $(pwd):/app -w /app gmtp
```
We recommend using docker for Pyserini's dependence on Java 21, but you may simply follow the code below.
```
conda create -n gmtp python=3.10
pip install -r requirements.txt
pip install -e ./beir
```

## 🔍 Usage
Convert Beir datasets into fixed format.
```
bash scripts/run_convert_dataset.sh
```
Convert poisoned datasets into fixed format.
- `attack`: Attack method (poisonedrag / phantom / advdecoding).
- `total`: Total amount of poisoned documents (For current setting, fix it to 200 as only 200 poisoned documents are provided.)
```
bash scripts/run_convert_poisoned_dataset.sh
```
Run Pyserini to index clean / attacked documents.
- `is_attack`: whether we are indexing clean or poisoned documents. 
You should run `is_attack=False` once for each `dataset`, and you should run `is_attack=True` once for every combination of `dataset` and `attack` to blend the poisoned documents into the clean ones.

- `dataset`: Target dataset (nq / hotpotqa / msmarco).
- `encoder_class`: Retriever (dpr / contriever).
```
bash scripts/run_faiss_indexing.sh
```
Now get average masked token probability of knowledge base.
- `retriever`: Retriever (dpr / contriever).
- `N` : Maximum amount of potential cheating tokens.
- `M` : The amount of tokens actually used for consideration.
- `total_samples`: The number of documents used for average gradient calculation ($K$ in paper)
- `use_random_doc`: Whether use random documents for single query or use relevant documents.
- `include_poison`: Whether to sample `total_samples` of documents from poisoned knowledge base.
```
bash scripts/run_get_avg_mask_probs.sh
```
Now we are ready to run GTMP against various attacks! Run the code below to reproduce the result. Otherwise, you may simply use code in `src/defenses/method/GMTP` for your work.
```
bash scripts/run_main.sh
```
- `retrieval_only`: Whether run only retrieval phase or until the generation phase.
- `latency_check`: Whether check latency.
- `debug`: Debug mode, using only 10 test samples.
- `api_key`: Input your OpenAI API key.
- `defense_method`: defense baselines (GMTP, PPL, L2).
- `reranker`: MLM of reranker. Can be either `bert` or `roberta`.
- `remove_threshold`: default remove threshold of GMTP. If higher than 0, it will use fixed remove threshold instead of calculated one from `run_get_avg_mask_probs.sh`.
- `use_random_doc`: Whether to use threshold calculated by random corpus or related corpus.
- `retrieve_k`: top-2k.
- `rerank_k`: top-k.



# References
- Our code used and contains [Beir](https://github.com/beir-cellar/beir) benchmark, with required library modification.
- Our code used [Pyserini](https://github.com/castorini/pyserini/tree/master).
- The base template was originated from [monologg](https://github.com/monologg).

# Citation
If you use this code, please cite the following our paper:
```
@inproceedings{kim2025safeguarding,
  title={Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection},
  author={Kim, San and Kim, Jonghwi and Jeon, Yejin and Lee, Gary},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={24597--24614},
  year={2025}
}
```
