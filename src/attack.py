import json
import random

import torch
import transformers
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRQuestionEncoder, LlamaForCausalLM

if transformers.__version__ != "4.40.1":
    from transformers import Gemma2ForCausalLM


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """

    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings
    if isinstance(model, DPRContextEncoder):
        embeddings = model.ctx_encoder.bert_model.embeddings.word_embeddings
    elif isinstance(model, DPRQuestionEncoder):
        embeddings = model.question_encoder.bert_model.embeddings.word_embeddings
    elif transformers.__version__ != "4.40.1" and isinstance(model, Gemma2ForCausalLM):
        embeddings = model.get_input_embeddings()
    elif isinstance(model, LlamaForCausalLM):
        embeddings = model.get_input_embeddings()
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings


def hotflip_attack(averaged_grad, embedding_matrix, increase_loss=False, num_candidates=1, filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(embedding_matrix, averaged_grad)
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


class Attacker:
    def __init__(self, conf, **kwargs) -> None:
        # assert args.attack_method in ['default', 'whitebox']

        self.attack_method = conf.attack.attack_method
        self.score_function = conf.attack.score_function
        self.adv_per_query = conf.attack.adv_per_query

        self.q_model = kwargs.get("q_model", None)
        self.c_model = kwargs.get("c_model", None)
        self.q_tokenizer = self.q_model.tokenizer
        self.c_tokenizer = self.c_model.tokenizer

        self.max_seq_length = kwargs.get("max_seq_length", 256)
        self.pad_to_max_length = kwargs.get("pad_to_max_length", True)
        self.per_gpu_eval_batch_size = kwargs.get("per_gpu_eval_batch_size", 64)
        self.num_adv_passage_tokens = kwargs.get("num_adv_passage_tokens", 30)

        self.num_cand = kwargs.get("num_cand", 100)
        self.num_iter = kwargs.get("num_iter", 30)
        self.gold_init = kwargs.get("gold_init", False)
        # self.early_stop = kwargs.get('early_stop', False) #  clean doc 중 top-1 score를 구한 뒤 hotflip 도중 score가 top-1 score보다 높아지면 stop.

        with open(conf.dataset.poisoned_text_path) as f:
            self.all_adv_texts = json.load(f)

    def get_attack(self, total=10) -> list:
        """Generate poisoend documents for "total" queries.
        "adv_per_query" number of poisoned documents are generated for each query.
        poisoned_docs (list) : list of poisoned documents, size of (total, adv_per_query)
        cheating_texts (list) : list of cheating texts, size of (total, adv_per_query)

        Args:
            total (int, optional): Number of target queries. Defaults to 10.

        Returns:
            target_attacks (list) : [{"question": str, "id": int, "poisoned_docs": list, "cheating_texts": list, etc.}]
        """
        poisoned_docs = []  # get the adv_text for the iter
        cheating_texts = []
        target_poisoned_texts = self.all_adv_texts[:total]
        if self.attack_method == "query_target":
            for i, data in enumerate(target_poisoned_texts):
                question = data["query"]
                adv_texts_b = data["poisoned_texts"][: self.adv_per_query]
                adv_text_a = question + "."
                adv_texts = [adv_text_a + i for i in adv_texts_b]
                adv_rets = [adv_text_a for i in range(len(adv_texts_b))]
                poisoned_docs.append(adv_texts)
                cheating_texts.append(adv_rets)
        # elif self.attack_method == "hotflip":
        else:
            poisoned_docs, cheating_texts = self.hotflip(target_poisoned_texts)

        for i in range(len(target_poisoned_texts)):
            target_poisoned_texts[i]["poisoned_docs"] = poisoned_docs[i]
            target_poisoned_texts[i]["cheating_texts"] = cheating_texts[i]

        return target_poisoned_texts

    def hotflip(self, target_poisoned_texts, adv_b=None, **kwargs) -> list:
        device = "cuda"
        print("Doing HotFlip attack!")
        adv_text_groups = []
        adv_ret_groups = []
        for data in tqdm(target_poisoned_texts):
            query = data["query"]
            # top1_score = query_score['top1_score']
            # adv_texts_b = self.all_adv_texts[id]["adv_texts"]
            adv_texts_b = data["poisoned_texts"]

            adv_texts = []
            ret_texts = []
            for j in range(self.adv_per_query):
                adv_b = adv_texts_b[j]
                adv_b = self.c_tokenizer(adv_b, max_length=self.max_seq_length, truncation=True, padding=False)[
                    "input_ids"
                ]
                if self.gold_init:
                    adv_a = query
                    adv_a = self.c_tokenizer(adv_a, max_length=self.max_seq_length, truncation=True, padding=False)[
                        "input_ids"
                    ]

                else:  # init adv passage using [MASK]
                    adv_a = [self.c_tokenizer.mask_token_id] * self.num_adv_passage_tokens

                embeddings = get_embeddings(self.c_model.model)  # embedding layer를 가져옴
                embedding_gradient = GradientStorage(embeddings)

                adv_passage = adv_a + adv_b  # token ids # retriever-preferred text + adv_text
                adv_passage_ids = torch.tensor(adv_passage, device=device).unsqueeze(0)
                adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
                adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

                # q_sent = self.tokenizer(query, max_length=self.max_seq_length, truncation=True, padding="max_length" if self.pad_to_max_length else False, return_tensors="pt")
                # q_sent = {key: value.cuda() for key, value in q_sent.items()}
                # q_emb = self.get_emb(self.q_model, q_sent).detach()
                q_emb = self.q_model.get_emb(
                    query, max_length=self.max_seq_length, padding="max_length" if self.pad_to_max_length else False
                ).detach()
                for it_ in range(self.num_iter):
                    grad = None
                    self.c_model.model.zero_grad()

                    p_sent = {
                        "input_ids": adv_passage_ids,
                        "attention_mask": adv_passage_attention,
                        "token_type_ids": adv_passage_token_type,
                    }
                    # p_emb = self.get_emb(self.c_model, p_sent)
                    p_emb = self.c_model.get_emb(inputs=p_sent)

                    if self.score_function == "dot":
                        sim = torch.mm(p_emb, q_emb.T)
                    elif self.score_function == "cos_sim":
                        sim = torch.cosine_similarity(p_emb, q_emb)
                    else:
                        raise KeyError

                    loss = sim.mean()
                    # if self.early_stop and sim.item() > top1_score + 0.1: break
                    loss.backward()

                    temp_grad = embedding_gradient.get()  # temp_grad : [1, seq_len, hidden_dim] ex) [1, 65, 768]
                    if grad is None:
                        grad = temp_grad.sum(dim=0)
                    else:
                        grad += temp_grad.sum(dim=0)

                    token_to_flip = random.randrange(
                        len(adv_a)
                    )  # adv_a + adv_b와 query의 sim을 보고, adv_a에 한해서 flip할 부분을 고르는 듯.
                    candidates = hotflip_attack(
                        grad[token_to_flip],
                        embeddings.weight,
                        increase_loss=True,
                        num_candidates=self.num_cand,
                        filter=None,
                    )
                    current_score = 0
                    candidate_scores = torch.zeros(self.num_cand, device=device)

                    temp_score = loss.sum().cpu().item()
                    current_score += temp_score

                    for i, candidate in enumerate(candidates):
                        temp_adv_passage = adv_passage_ids.clone()
                        temp_adv_passage[:, token_to_flip] = candidate  # 랜덤한 토큰을 candidate 중 하나로 변경해봄.
                        temp_p_sent = {
                            "input_ids": temp_adv_passage,
                            "attention_mask": adv_passage_attention,
                            "token_type_ids": adv_passage_token_type,
                        }
                        # temp_p_emb = self.get_emb(self.c_model, temp_p_sent)
                        temp_p_emb = self.c_model.get_emb(inputs=temp_p_sent)
                        with torch.no_grad():
                            if self.score_function == "dot":
                                temp_sim = torch.mm(temp_p_emb, q_emb.T)
                            elif self.score_function == "cos_sim":
                                temp_sim = torch.cosine_similarity(temp_p_emb, q_emb)
                            else:
                                raise KeyError
                            can_loss = temp_sim.mean()
                            temp_score = can_loss.sum().cpu().item()
                            candidate_scores[i] += temp_score

                    # if find a better one, update
                    if (candidate_scores > current_score).any():
                        best_candidate_idx = candidate_scores.argmax()
                        adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                    else:
                        continue
                ret_text = self.c_tokenizer.decode(
                    adv_passage_ids[0][: len(adv_a)], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                adv_text = self.c_tokenizer.decode(
                    adv_passage_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                adv_texts.append(adv_text)
                ret_texts.append(ret_text)
            adv_text_groups.append(adv_texts)
            adv_ret_groups.append(ret_texts)
        return adv_text_groups, adv_ret_groups
