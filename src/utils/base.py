import random
from datetime import datetime

import numpy as np
import torch

from src.retriever import DPR, Contriever


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_retrievers(conf):
    if conf.model.retriever == "contriever":
        q_model_name = "facebook/contriever-msmarco"
        c_model_name = "facebook/contriever-msmarco"
        model_class = Contriever
    elif conf.model.retriever == "dpr":
        q_model_name = "facebook/dpr-question_encoder-single-nq-base"
        c_model_name = "facebook/dpr-ctx_encoder-single-nq-base"
        model_class = DPR
    if conf.model.reranker == "bert":
        reranker_name = "bert-base-uncased"
    elif conf.model.reranker == "roberta":
        reranker_name = "FacebookAI/roberta-base"

    use_cuda = torch.cuda.is_available()
    c_model = model_class(c_model_name, use_cuda=use_cuda, is_question_encoder=False, reranker=reranker_name)
    if conf.model.retriever == "contriever":
        q_model = c_model
    else:
        q_model = model_class(q_model_name, use_cuda=use_cuda, is_question_encoder=True)

    return q_model, c_model


def search_poisoned_doc_by_query_id(poisoned_docs, query_id):
    for pos in poisoned_docs:
        if pos["query_id"] == query_id:
            return pos
    return None


def search_poisoned_doc_by_text(poisoned_docs, query_id, text):
    pos = search_poisoned_doc_by_query_id(poisoned_docs, query_id)

    for pos in poisoned_docs:
        if pos["query_id"] == query_id:
            for i, doc in enumerate(pos["poisoned_docs"]):
                if doc == text:
                    return {
                        "poisoned_text": pos["poisoned_texts"][i],
                        "poisoned_doc": doc,
                        "cheating_text": pos["cheating_texts"][i],
                    }
    return None


def get_formatted_time():
    now = datetime.now()
    return now.strftime("%m%d%H%M")
