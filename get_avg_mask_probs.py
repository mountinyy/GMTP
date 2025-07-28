import argparse
import json
import os
import random

import jsonlines
import numpy as np
import torch
from tqdm import tqdm

from src.attack import GradientStorage, get_embeddings
from src.retriever import DPR, Contriever
from src.utils.base import set_seed
from src.utils.beir_utils import load_dataset as load_beir_dataset


def load_retriever(args):
    if args.retriever == "contriever":
        q_model_name = "facebook/contriever-msmarco"
        c_model_name = "facebook/contriever-msmarco"
        model_class = Contriever
    elif args.retriever == "dpr":
        q_model_name = "facebook/dpr-question_encoder-single-nq-base"
        c_model_name = "facebook/dpr-ctx_encoder-single-nq-base"
        model_class = DPR
    if args.reranker == "bert":
        reranker_name = "bert-base-uncased"
    elif args.reranker == "roberta":
        reranker_name = "FacebookAI/roberta-base"
    use_cuda = torch.cuda.is_available()
    c_model = model_class(c_model_name, use_cuda=use_cuda, is_question_encoder=False, reranker=reranker_name)
    if args.retriever == "contriever":
        q_model = c_model
    else:
        q_model = model_class(q_model_name, use_cuda=use_cuda, is_question_encoder=True)

    return q_model, c_model


def main(args):
    corpus, queries, qrels = load_beir_dataset(args.dataset, "test" if args.dataset == "nq" else "train")

    q_model, c_model = load_retriever(args)
    embeddings = get_embeddings(c_model.model)
    embedding_gradient = GradientStorage(embeddings)

    masked_probs = []
    masked_probs_min = []
    masked_probs_mean = []

    save_root = f"data/mask_probs/{args.attack}"
    file_name = f"{args.dataset}_{args.retriever}_{args.reranker}_{args.N}_min{args.M}.json"
    if args.random_doc:
        save_path = os.path.join(save_root, "random_" + file_name)
    else:
        save_path = os.path.join(save_root, file_name)
        # random에 따른 savepath./
    os.makedirs(save_root, exist_ok=True)
    print(f"Saving to {save_path}")

    # target_dataset = filter_dataset(args, dataset, model.tokenizer)
    query_keys = list(queries.keys())
    random.shuffle(query_keys)
    pbar = tqdm(total=min(args.total_samples, len(query_keys)))
    finish = False
    for q_id in query_keys:
        q_model.model.zero_grad()
        c_model.model.zero_grad()
        query = queries[q_id]
        if args.random_doc:
            gt_doc_ids = random.sample(list(corpus.keys()), 1)
            gt_docs = [corpus[doc_id]["title"] + "\n" + corpus[doc_id]["text"].strip() for doc_id in gt_doc_ids]
        else:
            gt_doc_ids = list(qrels[q_id].keys())
            gt_docs = [corpus[doc_id]["title"] + "\n" + corpus[doc_id]["text"].strip() for doc_id in gt_doc_ids]
        q_emb = q_model.get_emb(query).detach()

        for doc in gt_docs:
            tok = c_model.tokenizer(doc, return_tensors="pt", max_length=512, truncation=True, add_special_tokens=False)
            if len(tok["input_ids"][0]) <= args.N:
                continue

            d_emb = c_model.get_emb(doc)
            sim = torch.mm(d_emb, q_emb.T)
            sim.backward()
            grad = embedding_gradient.get()
            scores = grad.norm(dim=-1)[0]
            top_indices = scores.topk(min(args.N, len(scores))).indices
            cur_masked_probs = c_model.get_mask_probs(doc, top_indices)
            if len(cur_masked_probs) == 0:
                continue
            cur_masked_probs = np.sort(cur_masked_probs)[: args.M]
            masked_probs.extend(cur_masked_probs)
            masked_probs_min.append(np.min(cur_masked_probs))
            masked_probs_mean.append(np.mean(cur_masked_probs))
            pbar.update(1)
            if len(masked_probs) >= args.total_samples * args.M:
                pbar.close()
                finish = True
                break
        if finish:
            break
    avg_min = np.mean([value for value in masked_probs_min if value < 0.01])
    print("avg_min :", avg_min)
    save_item = {
        "mean": np.mean(masked_probs),
        "std": np.std(masked_probs),
        "min": np.mean(masked_probs_min),
        "min_under_0.01": avg_min,
        "min_of_min": np.min(masked_probs_min),
        "min_of_mean": np.min(masked_probs_mean),
    }
    print("mean :", np.mean(masked_probs))
    print("std :", np.std(masked_probs))
    print("mean of min :", np.mean(masked_probs_min))
    print("mean min_under_0.01 :", avg_min)
    print("min_of_min :", np.min(masked_probs_min))
    print("min_of_mean :", np.min(masked_probs_mean))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_item, f)
    print(f"Saved to {save_path}")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get average mask probabilities")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--attack", type=str)
    parser.add_argument("--retriever", type=str)
    parser.add_argument("--reranker", type=str)
    parser.add_argument("--poison_doc_path", type=str)
    parser.add_argument("--N", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--M", type=int)
    parser.add_argument("--total_samples", type=int)
    parser.add_argument("--include_poison", action="store_true")
    parser.add_argument("--random_doc", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    main(args)
