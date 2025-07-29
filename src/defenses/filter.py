import json
import math
import os
import time
from collections import defaultdict

import ir_measures
import numpy as np
import torch
from ir_measures import nDCG
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.attack import GradientStorage, get_embeddings
from src.utils.base import search_poisoned_doc_by_query_id
from src.utils.defenses import get_poison_cnt


@torch.no_grad()
def get_skewed_attention_indices(
    model, query, text, layer_start, layer_end, return_attentions=False, threshold=3.0, local="mean"
):
    input = model.tokenizer(query, text, return_tensors="pt", padding=True, truncation=True)
    query_input = model.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    attention = model.get_attn(query, text, padding=True, truncation=True)  # [layer, head, seq, seq]
    query_text_attention = attention[
        :, :, 1 : query_input["input_ids"].size(-1) - 1, query_input["input_ids"].size(-1) :
    ]  # Consider only query part.
    attention_sum = query_text_attention.sum(dim=1)  # [layer, seq, seq]
    sep_pos = torch.where(input["input_ids"][0][query_input["input_ids"].size(-1) :] == model.tokenizer.sep_token_id)[
        0
    ].cpu()
    attention_sum[:, :, sep_pos] = 0
    attention_mean = attention_sum.mean(dim=-2)  # [layer, seq(K)]
    attention_mean = attention_mean[layer_start:layer_end].mean(dim=0)  # [seq(K)]
    if local == "mean":
        threshold = attention_mean.mean().item()
    elif local == "std":
        threshold = attention_mean.mean().item() + attention_mean.std().item()
    elif local == "std2":
        threshold = attention_mean.mean().item() + attention_mean.std().item() * 2
    output = (torch.where(attention_mean > threshold)[0] + 1).tolist()

    if return_attentions:
        return output, attention_mean
    return output


@torch.no_grad()
def get_hid_cross(c_model, q_model, text, query):
    c_hid = c_model.get_last_hidden_states(text)
    q_hid = q_model.get_last_hidden_states(query)
    output = torch.bmm(c_hid, q_hid.permute(0, -1, -2))
    output = output.sum(dim=-1)
    output = output.squeeze(0)

    return output.tolist()


@torch.no_grad()
def get_masked_sim(topk_indices, q_model, query, c_model, context, score_function="dot"):
    masked_emb, masked_text = c_model.get_emb(
        texts=context, mask_indices=topk_indices, return_text=True, padding=True, truncation=True
    )
    masked_emb = masked_emb.detach().cpu()
    query_emb = q_model.get_emb(texts=query).detach().cpu()
    if score_function == "dot":
        sim = torch.mm(masked_emb, query_emb.T).item()
    elif score_function == "cosine":
        sim = torch.cosine_similarity(masked_emb, query_emb).item()

    return sim, masked_text


def z_score_normalize(input_list):
    array = np.array(input_list)
    mean = np.mean(array)
    std = np.std(array)
    return (array - mean) / std


def z_score_normalize_sets(scores):
    all_scores, lengths = [], []
    for score in scores:
        all_scores.extend(score)
        lengths.append(len(score))
    all_scores = z_score_normalize(all_scores)
    normalized_scores = []
    start = 0
    for length in lengths:
        normalized_scores.append(all_scores[start : start + length])
        start += length
    return normalized_scores


def min_max_normalize(input_list):
    array = np.array(input_list)
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)


def get_probs(input_list):
    sum_value = sum(input_list)
    return [item / sum_value for item in input_list]


def min_max_normalization(data):
    if isinstance(data, list):
        data = torch.stack(data, dim=0)
    min_val = torch.min(data)
    max_val = torch.max(data)
    return (data - min_val) / (max_val - min_val)


@torch.no_grad()
def calculate_magnitude_and_angle_differences(question, embeddings):
    # Stack the list of embedding tensors into a single tensor
    embeddings = torch.stack(embeddings, dim=0).squeeze(1)

    # Calculate norms of question and embeddings
    norm_type = 2  # fro nuc 2 1
    question_norm = torch.norm(question, p=norm_type)
    embeddings_norms = torch.norm(embeddings, dim=1, p=norm_type)

    # Calculate magnitude differences
    abs_differences = torch.abs(embeddings_norms - question_norm)  # Absolute differences
    ratio_differences = embeddings_norms / question_norm  # Ratio of magnitudes

    # Normalize question and embeddings to compute angles
    question_normalized = question / question_norm
    embeddings_normalized = embeddings / embeddings_norms.unsqueeze(1)
    # Compute cosine similarities and angles
    cos_angles = torch.clamp(torch.matmul(embeddings_normalized, question_normalized.T), -1.0, 1.0)
    angles = torch.acos(cos_angles)  # Angles in radians

    # Calculate averages
    avg_abs_difference = abs_differences.mean().item()
    avg_ratio = ratio_differences.mean().item()
    avg_angle_radians = angles.mean().item()
    avg_angle_degrees = np.degrees(avg_angle_radians)
    # avg_angle_degrees = torch.degrees(angles).mean().item()

    return avg_abs_difference, avg_ratio, avg_angle_radians, avg_angle_degrees


def get_norm_diffs(question, embeddings, norm_type=2):
    if isinstance(embeddings, list):
        embeddings = torch.stack(embeddings, dim=0).squeeze(1)
    question_norm = torch.norm(question, p=norm_type)
    embeddings_norms = torch.norm(embeddings, dim=1, p=norm_type)
    abs_differences = torch.abs(embeddings_norms - question_norm)

    return abs_differences


def apply_normalization(data, idx, type="min_max"):
    data_list = [item[idx] for item in data]
    if type == "min_max":
        normalized = min_max_normalize(data_list)
    elif type == "z_score":
        normalized = z_score_normalize(data_list)
    for i in range(len(data)):
        data[i][idx] = normalized[i]
    return data


def get_ppl(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], labels=inputs["input_ids"])
        loss = outputs.loss
    return math.exp(loss.item())


def apply(conf, q_model, c_model, corpus, p_corpus, poison_infos, retrieval_results, qrels, return_output_text=False):
    poison_recalls = []
    return_outputs = []
    pos_in_topk = 0
    pos_in_topk_before = 0
    pos_in_top2k = 0
    gt_in_topk = 0
    gt_in_top2k = 0
    gt_in_topk_before = 0
    total_gt_means, total_poison_means = [], []
    latency_list = []
    leftover_cnt = 0
    leftovers = []
    gt_eval = {"total_cnt": 0, "removed_cnt": 0}
    removed_doc_cnt = 0
    gt_ranks_before = []
    gt_ranks = []
    gradient_scores = defaultdict(dict)
    gradient_scores["poison"] = defaultdict(list)
    gradient_scores["gt"] = defaultdict(list)
    gradient_scores["clean"] = defaultdict(list)
    diff_ratios = defaultdict(list)
    diff_ratios["poison"] = defaultdict(list)
    diff_ratios["gt"] = defaultdict(list)
    diff_ratios["clean"] = defaultdict(list)
    removed_ratios = defaultdict(list)
    total_poison_cnt_after, total_poison_cnt_before, total_poison_cnt_2k = 0, 0, 0
    embs = defaultdict(list)
    l2_norms = defaultdict(list)
    ppls = defaultdict(list)
    masked_texts = {}
    masked_texts["poison"] = defaultdict(list)
    masked_texts["gt"] = defaultdict(list)
    masked_texts["clean"] = defaultdict(list)
    metric_data = defaultdict(list)
    removed_docs = []

    if conf.defense.method == "perplexity":
        model_name = "openai-community/gpt2"
        gpt = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    print()
    ndcg_data = defaultdict(list)
    threshold_path = (
        os.path.join(
            os.path.dirname(conf.common.remove_threshold_path),
            "random_" + os.path.basename(conf.common.remove_threshold_path),
        )
        if conf.common.remove_threshold_random_doc
        else conf.common.remove_threshold_path
    )
    if conf.common.remove_threshold > 0:
        print("Using fixed remove threshold: ", conf.common.remove_threshold, flush=True)
        conf.common.remove_lambda = 1.0
    elif os.path.exists(threshold_path):
        print("Load remove threshold from", threshold_path, flush=True)
        with open(threshold_path, "r", encoding="utf-8") as f:
            conf.common.remove_threshold = json.load(f)["mean"]
    else:
        print("Failed to load remove threshold from", conf.common.remove_threshold_path, flush=True)
    print(
        f"Setting remove threshold to {conf.common.remove_threshold} * {conf.common.remove_lambda} -> {conf.common.remove_threshold * conf.common.remove_lambda}",
        flush=True,
    )
    print()

    embeddings = get_embeddings(c_model.model)
    embedding_gradient = GradientStorage(embeddings)

    for ret in tqdm(retrieval_results, total=len(retrieval_results)):
        query_id = ret["query_id"]
        poison_info = search_poisoned_doc_by_query_id(poison_infos, query_id)
        doc_infos = []
        question = poison_info["query"]
        for doc_id, score in ret["topk"]:
            is_poisoned = "poison" in doc_id
            if is_poisoned:
                doc_infos.append((doc_id, p_corpus[doc_id], score))
            else:
                doc_infos.append((doc_id, corpus[doc_id], score))

        ret_scores = []
        start_time = time.perf_counter()
        torch.cuda.synchronize()
        if conf.defense.method == "gmtp":
            for doc_id, doc, score in doc_infos:
                c_model.model.zero_grad()
                q_model.model.zero_grad()
                q_input = q_model.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(
                    q_model.model.device
                )
                c_input = c_model.tokenizer(doc, return_tensors="pt", padding=True, truncation=True).to(
                    c_model.model.device
                )
                cur_q_emb = q_model.get_emb(inputs=q_input).detach()
                emb = c_model.get_emb(inputs=c_input)
                sim = torch.mm(emb, cur_q_emb.T)
                sim.backward()

                grad = embedding_gradient.get()
                scores = grad.norm(dim=-1)

                ret_scores.append(scores[0].tolist())

                if "poison" in doc_id:
                    target = "poison"
                elif doc_id in poison_info["gt_id"]:
                    target = "gt"
                else:
                    target = "clean"
                gradient_scores[target]["scores"].append(scores[0].tolist())
                gradient_scores[target]["score_avg"].append(np.mean(scores[0].tolist()))
                gradient_scores[target]["score_max"].append(max(scores[0].tolist()))
                gradient_scores[target]["score_min"].append(min(scores[0].tolist()))

                del cur_q_emb
            total_scores = []
            for scores in ret_scores:
                total_scores.extend(scores)

            threshold = np.mean(total_scores)
        docs = []
        poison_mins = []
        gt_mins = []
        for j, (doc_id, doc, score) in enumerate(doc_infos):
            if "poison" in doc_id:
                target = "poison"
            elif doc_id in poison_info["gt_id"]:
                target = "gt"
            else:
                target = "clean"

            with torch.no_grad():
                emb = c_model.get_emb(texts=doc).detach().cpu()
            if conf.defense.method == "gmtp":
                grad_scores = torch.Tensor(ret_scores[j])
                threshold = grad_scores.mean().item()
                selected_indices = torch.where(grad_scores > threshold)[0]
                selected_values = grad_scores[selected_indices]
                top_indices = selected_values.topk(min(conf.common.N, len(selected_values))).indices
                top_indices = selected_indices[top_indices]

                # mask-prob
                if len(top_indices) == 0:
                    masked_probs = [0]

                else:
                    with torch.no_grad():
                        masked_probs = c_model.get_mask_probs(text=doc, mask_indices=top_indices)
                        gradient_scores[target]["masked_probs_all"].extend(masked_probs)
                        masked_probs = np.sort(masked_probs)
                        masked_probs = masked_probs[: conf.common.M]

                gradient_scores[target]["masked_probs"].extend(masked_probs)
                gradient_scores[target]["masked_probs_avg"].append(np.mean(masked_probs))
                gt_ids = poison_info["gt_id"]
                if "poison" in doc_id:
                    poison_mins.append(min(masked_probs))
                    total_poison_means.append(np.mean(masked_probs))
                elif doc_id in gt_ids:
                    gt_mins.append(min(masked_probs))
                    total_gt_means.append(np.mean(masked_probs))

                # mask by top_indices

                if len(top_indices) == 0:
                    sim = score
                    masked_text = doc
                else:
                    sim, masked_text = get_masked_sim(
                        top_indices, q_model, question, c_model, doc, score_function="dot"
                    )
                sim_diff = abs(score - sim)

                saving_masked_prob = np.mean(masked_probs)
                metric_data[target].append(saving_masked_prob)

                docs.append([doc_id, doc, score, sim, masked_text, sim_diff, top_indices, emb, saving_masked_prob])

            elif conf.defense.method == "perplexity":
                ppl = get_ppl(gpt, tokenizer, doc)
                metric_data[target].append(ppl)
                ppls[target].append(ppl)
                docs.append([doc_id, doc, score, ppl])

            elif conf.defense.method == "l2":
                l2_norm = torch.norm(emb, p=2).item()
                metric_data[target].append(l2_norm)
                l2_norms[target].append(l2_norm)
                docs.append([doc_id, doc, score, l2_norm])

            embs[target].append(emb)

        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000
        latency_list.append(latency)
        docs = sorted(docs, key=lambda x: x[2], reverse=True)

        for gt_id in poison_info["gt_id"]:
            if gt_id in [doc[0] for doc in docs[: conf.rag.top_k]]:
                gt_in_topk_before += 1
                break

        for j, item in enumerate(docs[: conf.rag.top_k]):
            if "poison" in item[0]:
                pos_in_topk_before += 1
                break
        for j, item in enumerate(docs):
            for gt_id in poison_info["gt_id"]:
                if gt_id == item[0]:
                    gt_ranks_before.append(j + 1)
                    break

        # Get data for nDCG
        for gt_id in poison_info["gt_id"]:
            rel_score = qrels[query_id][gt_id]
            ndcg_data["qrels"].append(ir_measures.Qrel(query_id, gt_id, rel_score))
        for doc in docs[: conf.rag.top_k]:
            ndcg_data["before_result"].append(ir_measures.ScoredDoc(query_id, doc[0], doc[2]))

        poison_cnt_before = get_poison_cnt([doc[0] for doc in docs[: conf.rag.top_k]], poison_info, p_corpus)
        total_poison_cnt_before += poison_cnt_before

        # Mask Probs
        if conf.defense.method == "gmtp":
            threshold = conf.common.remove_threshold * conf.common.remove_lambda
            filtered_docs = [doc for doc in docs if doc[-1] > threshold]
            removed_docs = [doc for doc in docs if doc[-1] <= threshold]
            topk_removed_docs = [doc for doc in docs[: conf.rag.top_k] if doc[-1] <= threshold]
            removed_doc_cnt += len(topk_removed_docs)
            removed_ratios["total"].append((len(docs) - len(filtered_docs)) / len(docs))

        elif conf.defense.method == "perplexity":
            threshold = conf.perplexity.threshold
            filtered_docs = [doc for doc in docs if doc[-1] < threshold]
        elif conf.defense.method == "paraphrased":
            filtered_docs = docs
        elif conf.defense.method == "l2":
            threshold = conf.l2.contriever_threshold if conf.model.retriever == "contriever" else conf.l2.dpr_threshold
            filtered_docs = [doc for doc in docs if doc[-1] < threshold]

        topk_docs = filtered_docs[: conf.rag.top_k]
        for doc in topk_docs:
            ndcg_data["after_result"].append(ir_measures.ScoredDoc(query_id, doc[0], doc[2]))
        if conf.rag.top_k > len(topk_docs):
            leftover_cnt += 1
            leftovers.append(conf.rag.top_k - len(topk_docs))
        for i in range(conf.rag.top_k - len(topk_docs)):
            ndcg_data["after_result"].append(ir_measures.ScoredDoc(query_id, f"dummy{i}", 0.0))

        # Analyse
        # 1. Poison in docs
        poison_cnt = get_poison_cnt([doc[0] for doc in topk_docs], poison_info, p_corpus)
        poison_recalls.append(poison_cnt / conf.attack.adv_per_query)
        pos_in_topk += 1 if poison_cnt > 0 else 0
        poison_cnt_2k = get_poison_cnt([doc[0] for doc in docs], poison_info, p_corpus)
        pos_in_top2k += 1 if poison_cnt_2k > 0 else 0
        total_poison_cnt_after += poison_cnt
        total_poison_cnt_2k += poison_cnt_2k

        # 2. GT in docs
        for gt_id in poison_info["gt_id"]:
            if gt_id in [doc[0] for doc in docs]:
                gt_in_top2k += 1
                break
        for gt_id in poison_info["gt_id"]:
            if gt_id in [doc[0] for doc in topk_docs]:
                gt_in_topk += 1
                break
        for j, item in enumerate(docs):
            for gt_id in poison_info["gt_id"]:
                if gt_id == item[0]:
                    gt_ranks.append(j + 1)
                    break
        for gt_id in poison_info["gt_id"]:
            if gt_id in [doc[0] for doc in removed_docs]:
                gt_eval["removed_cnt"] += 1
            gt_eval["total_cnt"] += 1

        return_outputs.append(
            {"query_id": query_id, "topk": [[item[0], item[2]] for item in topk_docs], "topk_docs": topk_docs}
        )

    if len(gradient_scores["poison"]["masked_probs"]) == 0:
        gradient_scores["poison"]["masked_probs"].append(0)
        gradient_scores["poison"]["masked_probs_avg"].append(0)

    ndcg_before = ir_measures.calc_aggregate([nDCG @ 10], ndcg_data["qrels"], ndcg_data["before_result"])[nDCG @ 10]
    ndcg_after = ir_measures.calc_aggregate([nDCG @ 10], ndcg_data["qrels"], ndcg_data["after_result"])[nDCG @ 10]
    outputs = ""
    outputs += f"[Attack: {conf.attack.name}, Method: {conf.attack.attack_method} Dataset: {conf.dataset.name}, Retriever: {conf.model.retriever}, N: {conf.common.N}, M: {conf.common.M},  reranker: {conf.model.reranker}]\n"
    outputs += f"[Defense: {conf.defense.method}]\n"
    if removed_doc_cnt > 0:
        outputs += (
            f"GT FPR: {gt_eval['removed_cnt']/removed_doc_cnt:.3f} ({gt_eval['removed_cnt']} / {removed_doc_cnt})\n"
        )
    else:
        outputs += "GT FPR: No GT removed\n"
    if conf.defense.method == "perplexity":
        outputs += f"Perplexity threshold: {conf.perplexity.threshold}\n"
    outputs += f"Remove threshold: {conf.common.remove_threshold:.3f}\n"
    outputs += f"Remove lambda: {conf.common.remove_lambda}\n"
    outputs += f"nDCG@{conf.rag.top_k} : {ndcg_before:.3f} (clean), {ndcg_after:.3f} (after)\n"
    if total_poison_cnt_before == 0:
        outputs += f"Filtering rate : No poison detected in topk(before). after : {total_poison_cnt_after}\n"
    else:
        outputs += f"Filtering rate : {(total_poison_cnt_before - total_poison_cnt_after)/total_poison_cnt_before:.3f} ({(total_poison_cnt_before - total_poison_cnt_after)}/{total_poison_cnt_before})\n"
    outputs += f"Poison in topk (clean): {pos_in_topk_before/len(retrieval_results):.3f} ({pos_in_topk_before}/{len(retrieval_results)})\n"
    outputs += f"Poison in topk (poisoned): {pos_in_topk/len(retrieval_results):.3f} ({pos_in_topk}/{len(retrieval_results)})\n"
    outputs += "-" * 50 + "\n"
    if len(diff_ratios["poison"]["mean"]) > 0:
        outputs += f"Poison diff ratio : {np.mean(diff_ratios['poison']['mean']):.3f} (mean), , {np.min(diff_ratios['poison']['min']):.3f} (min), {np.mean(diff_ratios['poison']['min']):.3f} (min_avg)\n"
    if conf.defense.method == "gmtp":
        outputs += f"Masked probs avg. (N) : {np.mean(gradient_scores['poison']['masked_probs_all'])} (poison), {np.mean(gradient_scores['gt']['masked_probs_all'])} (gt), {np.mean(gradient_scores['clean']['masked_probs_all'])} (clean)\n"
        outputs += f"Masked probs avg. (M) : {np.mean(gradient_scores['poison']['masked_probs'])} (poison), {np.mean(gradient_scores['gt']['masked_probs'])} (gt), {np.mean(gradient_scores['clean']['masked_probs'])} (clean)\n"
    if conf.defense.method == "l2":
        outputs += f"L2 norm : {np.mean(l2_norms['poison']):.3f} (poison), {np.mean(l2_norms['gt']):.3f} (gt), {np.mean(l2_norms['clean']):.3f} (clean)\n"
    elif conf.defense.method == "perplexity":
        outputs += f"Perplexity : {np.mean(ppls['poison']):.3f} (poison), {np.mean(ppls['gt']):.3f} (gt), {np.mean(ppls['clean']):.3f} (clean)\n"
    outputs += "-" * 50 + "\n"

    outputs += "\n[Retrieval Latency]\n"
    outputs += f"Latency: {np.sum(latency_list):.3f} ms (sum), {np.mean(latency_list):.3f} ms (avg), {max(latency_list):.3f} ms (max), {min(latency_list):.3f} ms (min)\n"

    if return_output_text:
        return return_outputs, outputs
    return return_outputs
