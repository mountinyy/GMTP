import json
import logging
import os
import random
import time
from collections import defaultdict
from datetime import datetime

import jsonlines
from openai import OpenAI
from tqdm import tqdm

from src.defenses.filter import apply
from src.defenses.method import GMTP
from src.evaluation import gpt4_measure_by_answer, print_gpt4_eval_results
from src.generator import Generator
from src.utils.base import load_retrievers, search_poisoned_doc_by_query_id, search_poisoned_doc_by_text, set_seed
from src.utils.beir_utils import load_dataset as load_beir_dataset


def load_dataset(conf):
    corpus_path = os.path.join("data/beir", conf.dataset.name, "corpus.jsonl")
    with jsonlines.open(corpus_path) as f:
        corpus = {item["id"]: item["contents"] for item in f.iter()}
    print()
    print("##### Load dataset #####")
    if conf.common.do_attack:
        if conf.attack.name == "phantom":
            p_corpus_path = os.path.join(
                "data/poisoned_documents/",
                conf.attack.name,
                conf.attack.attack_method,
                conf.model.retriever,
                conf.model.generator,
                conf.dataset.name,
                "corpus.jsonl",
            )
            root, filename = os.path.split(conf.dataset.poisoned_doc_path)
            conf.dataset.poisoned_doc_path = os.path.join(root, conf.model.generator, filename)
        elif conf.attack.num_adv_passage_tokens < 10:
            p_corpus_path = os.path.join(
                "data/poisoned_documents/",
                conf.attack.name,
                conf.attack.attack_method,
                conf.model.retriever,
                f"{conf.dataset.name}-{conf.attack.num_adv_passage_tokens}",
                "corpus.jsonl",
            )
        else:
            p_corpus_path = os.path.join(
                "data/poisoned_documents/",
                conf.attack.name,
                conf.attack.attack_method,
                conf.model.retriever,
                conf.dataset.name,
                "corpus.jsonl",
            )
        with jsonlines.open(p_corpus_path) as f:
            p_corpus = {item["id"]: item["contents"] for item in f.iter()}
        print(f"Load p_corpus from {p_corpus_path}")
    else:
        if conf.attack.attack_method == "augmented":
            conf.dataset.poisoned_doc_path = conf.dataset.poisoned_doc_path.replace("hotflip", "augmented")
        p_corpus = {}
    print("poiseon_doc_path: ", conf.dataset.poisoned_doc_path)
    print("retrieval_result_path:", conf.rag.retrieval_result_path)
    with open(conf.dataset.poisoned_doc_path, "r", encoding="utf-8") as f:
        poison_info = json.load(f)
    with open(conf.rag.retrieval_result_path, "r", encoding="utf-8") as f:
        retrieval_result = json.load(f)
    return corpus, p_corpus, poison_info, retrieval_result


def main(conf):
    set_seed(conf.common.seed)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("fvcore").setLevel(logging.ERROR)
    logging.basicConfig(level=logging.WARNING)

    if conf.attack.name == "phantom" and conf.common.do_attack is False:
        root, filename = os.path.split(conf.dataset.poisoned_doc_path)
        root = os.path.join(root, conf.model.generator)
        conf.dataset.poisoned_doc_path = os.path.join(root, filename)

    generator = Generator(conf)

    corpus, p_corpus, poison_info, retrieval_result = load_dataset(conf)
    _, _, qrels = load_beir_dataset(conf.dataset.name, "dev" if conf.dataset.name == "msmarco" else "test")

    # Add qrels for augmented.
    if conf.attack.attack_method == "augmented":
        if conf.attack.name == "phantom":
            q_path = os.path.join(
                "data/poisoned_documents",
                conf.attack.name,
                conf.attack.attack_method,
                conf.model.retriever,
                conf.model.generator,
                f"{conf.dataset.name}-200.json",
            )
        else:
            q_path = os.path.join(
                "data/poisoned_documents",
                conf.attack.name,
                conf.attack.attack_method,
                conf.model.retriever,
                f"{conf.dataset.name}-200.json",
            )
        q_data = json.load(open(q_path, "r", encoding="utf-8"))
        for item in q_data:
            for gt_id in item["gt_id"]:
                qrels[item["query_id"]] = {gt_id: 1}

    q_model, c_model = load_retrievers(conf)

    if conf.common.debug:
        retrieval_result = retrieval_result[:10]

    # Retrieval Defense
    incorrect_answers = []
    correct_answers = []

    start_time = time.perf_counter()
    questions, total_clean_docs, total_gt_docs, total_poisoned_docs, responses = [], [], [], [], []
    topk_results, ret_output_text = apply(
        conf,
        q_model,
        c_model,
        corpus,
        p_corpus,
        poison_info,
        retrieval_result,
        qrels=qrels,
        return_output_text=True,
    )
    if conf.common.retrieval_only:
        print()
        print(ret_output_text, flush=True)
        exit()

    if not conf.common.do_attack:
        topk_results = retrieval_result

    c_model.model.to("cpu")
    q_model.model.to("cpu")
    gt_cnt = 0
    poison_cnt = 0
    poisoned_retrievals = 0
    ndcg_data = defaultdict(list)
    for ret in tqdm(topk_results, total=len(topk_results)):
        query_id = ret["query_id"]
        clean_docs, poison_docs, gt_docs = [], [], []
        poison_infos = []
        for doc_id, score in ret["topk"]:
            is_poisoned = "poison" in doc_id
            if is_poisoned:
                poison_docs.append(p_corpus[doc_id])
                poison_infos.append(search_poisoned_doc_by_text(poison_info, query_id, p_corpus[doc_id]))
                poison_cnt += 1
            else:
                clean_docs.append(corpus[doc_id])
        info = search_poisoned_doc_by_query_id(poison_info, query_id)
        question = info["query"]
        gt_docs = info["gt_text"]
        is_gt = False
        for gt_doc in gt_docs:
            for clean_doc in clean_docs:
                if gt_doc in clean_doc or gt_doc in clean_doc.split("\n")[1]:
                    is_gt = True
                    break
        docs = clean_docs + poison_docs

        if len(poison_docs) > 0:
            poisoned_retrievals += 1
        gt_found = False
        for gt_doc in gt_docs:
            for doc in clean_docs:
                if gt_doc in doc or doc in gt_doc:
                    gt_cnt += 1
                    gt_found = True
                    break
            if gt_found:
                break

        random.shuffle(docs)
        response = generator.generate(question, docs, use_priority=False, is_gt=is_gt)

        questions.append(question)
        responses.append(response)
        total_clean_docs.append([item for item in clean_docs])
        total_gt_docs.append([item for item in gt_docs])

        incorrect_answers.append(info["incorrect_answer"])
        correct_answers.append(info["correct_answer"])
    print(
        f"Poisoned retrievals : {poisoned_retrievals / len(topk_results):.3f} ({poisoned_retrievals}/{len(topk_results)})",
        flush=True,
    )

    gen_output = ""
    additional_names = ["-clean", "-poison"] if conf.common.do_attack else ["-clean"]
    client = OpenAI(api_key=conf.api_key)
    gpt4_measure_by_answer(
        questions,
        correct_answers,
        responses,
        conf.answer_following_prompt,
        conf.eval.save_path,
        client,
        additional_name="-clean",
    )
    if conf.common.do_attack:
        if conf.attack.name in ["phantom", "advdecoding"]:
            additional_names = ["-clean"]
            asr_cnt = 0
            refusals = ["cannot answer", "can't answer"]
            for j, res in enumerate(responses):
                found = False
                for refusal in refusals:
                    if refusal in res:
                        asr_cnt += 1
                        found = True
                        break
            gen_output += f"ASR: {asr_cnt / len(responses):.3f}\n"
        else:
            gpt4_measure_by_answer(
                questions,
                incorrect_answers,
                responses,
                conf.answer_following_prompt,
                conf.eval.save_path,
                client,
                additional_name="-poison",
            )
    if conf.common.do_attack and conf.attack.name == "poisonedrag":
        additional_names = ["-clean", "-poison"]
    else:
        additional_names = ["-clean"]
    gpt_outputs, answers = print_gpt4_eval_results(
        conf.eval.save_path, client, batch_id=None, max_try=180, additional_names=additional_names
    )

    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000

    print("=" * 50, flush=True)
    print(
        f"[Attack: {conf.attack.name} | Method: {conf.attack.attack_method} | Defense: {conf.defense.method}]",
        flush=True,
    )
    print(
        f"[Dataset: {conf.dataset.name} | Retriever: {conf.model.retriever} | Generator: {conf.model.generator}]",
        flush=True,
    )
    if ret_output_text:
        print("[Retrieval Analysis]", flush=True)
        print(ret_output_text, flush=True)

    print(flush=True)
    print(flush=True)
    print("[Generation Analysis]", flush=True)
    print(gen_output, flush=True)
    print(gpt_outputs, flush=True)
    if conf.common.latency_check:
        print("Defense method: ", conf.defense.method, flush=True)
        print(f"Overall Latency: {latency:.3f} ms", flush=True)
        if ret_output_text:
            print(flush=True)
            print("[Retrieval Analysis]", flush=True)
            print(ret_output_text, flush=True)

    save_item = {
        "questions": questions,
        "responses": responses,
        "correct_answers": correct_answers,
        "evaluations": answers,
        "incorrect_answers": incorrect_answers,
    }

    current_time = datetime.now().strftime("%m%d%H%M")
    save_path = os.path.join(conf.rag.result_save_path, f"{conf.defense.method}-{conf.rag.top_k}-{current_time}.json")
    if not conf.common.do_attack:
        save_path = save_path.replace(".json", "-no-attack.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_item, f, indent=4)
    print("Result saved at ", save_path, flush=True)


if __name__ == "__main__":
    main()
