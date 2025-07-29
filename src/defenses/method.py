from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
)


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
    if isinstance(model, DPRContextEncoder):
        embeddings = model.ctx_encoder.bert_model.embeddings.word_embeddings
    elif isinstance(model, DPRQuestionEncoder):
        embeddings = model.question_encoder.bert_model.embeddings.word_embeddings
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings


class GMTP:
    def __init__(
        self,
        ret_type: str,
        model_name: str,
        question_encoder_name: Optional[str] = None,
        reranker: str = "bert-base-uncased",
        N: int = 10,
        M: int = 5,
        remove_threshold: float = -1.0,
        remove_lambda: float = 1.0,
    ):
        assert ret_type in ["dpr", "contriever"], "GMTP is only supported for DPR and Contriever retrievers."
        # Retriever setup
        self.ret_type = ret_type
        if self.ret_type == "dpr":
            self.d_encoder = DPRContextEncoder.from_pretrained(model_name).to("cuda")
            self.d_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_name)
            self.q_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_name).to("cuda")
            self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(question_encoder_name)
        elif self.ret_type == "contriever":
            self.d_encoder = AutoModel.from_pretrained(model_name).to("cuda")
            self.d_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.q_encoder = self.d_encoder
            self.q_tokenizer = self.d_tokenizer
        self.d_tokenizer.model_max_length = 512
        self.q_tokenizer.model_max_length = 512

        # Reranker setup
        self.reranker = AutoModelForMaskedLM.from_pretrained(reranker).eval().to(self.d_encoder.device)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker)
        self.use_convert = self.d_tokenizer.get_vocab() != self.reranker_tokenizer.get_vocab()
        if self.use_convert:
            print("Using different vocabularies for Contriever and Reranker tokenizers.", flush=True)

        # Gradient setup
        embeddings = get_embeddings(self.d_encoder)
        self.embedding_gradient = GradientStorage(embeddings)

        # Others
        self.N = N
        self.M = M
        self.remove_threshold = remove_threshold
        self.remove_lambda = remove_lambda

        # Print
        print("=" * 10 + " GMTP Config " + "=" * 10)
        print(f"Retriever type: {self.ret_type}")
        print(f"Document encoder: {model_name}")
        if question_encoder_name is None:
            print(f"Question encoder: {model_name}")
        else:
            print(f"Question encoder: {question_encoder_name}")
        print(f"Reranker: {reranker}")
        print(f"Use convert: {self.use_convert}")
        print(f"N: {self.N}")
        print(f"M: {self.M}")
        print(f"Remove threshold: {self.remove_threshold}")
        print(f"Remove lambda: {self.remove_lambda}")
        print("=" * 30)

    def filter_documents(self, question, documents, topk=10, do_sort=False, doc_ids=None):
        doc_infos = []
        for i, doc in enumerate(documents):
            info = {"document": doc}
            if doc_ids is not None:
                info["id"] = doc_ids[i]
            self.d_encoder.zero_grad()
            self.q_encoder.zero_grad()
            q_input = self.q_tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(
                self.q_encoder.device
            )
            d_input = self.d_tokenizer(doc, return_tensors="pt", padding=True, truncation=True).to(
                self.d_encoder.device
            )
            cur_q_emb = self.get_emb(inputs=q_input, is_document_encoder=False).detach()
            emb = self.get_emb(inputs=d_input, is_document_encoder=True)
            sim = torch.mm(emb, cur_q_emb.T)
            sim.backward()
            info["sim"] = sim.item()
            grad = self.embedding_gradient.get()
            scores = grad.norm(dim=-1)
            info["ret_scores"] = scores[0].tolist()
            doc_infos.append(info)
            del cur_q_emb

        for j, doc in enumerate(documents):
            info = doc_infos[j]
            grad_scores = torch.Tensor(info["ret_scores"])
            threshold = grad_scores.mean().item()
            selected_indices = torch.where(grad_scores > threshold)[0]
            selected_values = grad_scores[selected_indices]
            top_indices = selected_values.topk(min(self.N, len(selected_values))).indices
            top_indices = selected_indices[top_indices]

            if len(top_indices) == 0:
                masked_probs = [0]
            else:
                with torch.no_grad():
                    masked_probs = self.get_mask_probs(text=doc, mask_indices=top_indices)
                    masked_probs = np.sort(masked_probs)
                    masked_probs = masked_probs[: self.M]

            if len(top_indices) == 0:
                sim = info["sim"]
            else:
                sim, _ = self.get_masked_sim(
                    top_indices, self.q_encoder, question, self.d_encoder, doc, score_function="dot"
                )

            avg_masked_prob = np.mean(masked_probs)
            doc_infos[j]["avg_masked_prob"] = avg_masked_prob

        sim_sorted_doc_infos = sorted(doc_infos, key=lambda x: x["sim"], reverse=True) if do_sort else doc_infos

        threshold = self.remove_threshold * self.remove_lambda
        filtered_docs = [info for info in sim_sorted_doc_infos if info["avg_masked_prob"] > threshold]
        topk_docs = [info["document"] for info in filtered_docs[:topk]]
        if doc_ids is not None:
            return topk_docs, [info["id"] for info in filtered_docs[:topk]]
        return topk_docs

    @torch.no_grad()
    def get_masked_sim(self, topk_indices, q_model, query, c_model, context, score_function="dot"):
        masked_emb, masked_text = self.get_emb(
            texts=context,
            mask_indices=topk_indices,
            return_text=True,
            padding=True,
            truncation=True,
            is_document_encoder=True,
        )
        masked_emb = masked_emb.detach().cpu()
        query_emb = self.get_emb(texts=query, is_document_encoder=False).detach().cpu()
        if score_function == "dot":
            sim = torch.mm(masked_emb, query_emb.T).item()
        elif score_function == "cosine":
            sim = torch.cosine_similarity(masked_emb, query_emb).item()

        return sim, masked_text

    def convert_tokens(self, text, mask_indices, max_length=512):
        # Tokenize the text with both tokenizers, asking for offset mappings
        encoding_a = self.d_tokenizer(text, return_offsets_mapping=True)
        encoding_b = self.reranker_tokenizer(text, return_offsets_mapping=True)

        offsets_a = encoding_a["offset_mapping"]  # List of (start, end) for each token in A
        offsets_b = encoding_b["offset_mapping"]  # List of (start, end) for each token in B

        # This will hold, for each A-index, the list of B-indices that overlap
        a_to_b_indices = []

        # We can keep a pointer `j` into B-tokens to speed things up,
        # but it is not strictly required. We'll do it to avoid re-scanning from 0 each time.
        j = 0
        b_len = len(offsets_b)

        # Ensure the A-indices are in ascending order (so that the pointer approach works best):
        a_indices_sorted = sorted(mask_indices)

        for a_idx in a_indices_sorted:
            # Get start/end character offsets for this A-token
            a_start, a_end = offsets_a[a_idx]

            # Advance `j` if B-tokens end before `a_start` (no overlap).
            while j < b_len and offsets_b[j][1] <= a_start:
                j += 1

            # Now collect all B tokens that overlap [a_start, a_end).
            # Specifically, we want all tokens where b_start < a_end
            # and b_end > a_start, but usually we check b_start < a_end
            # assuming we already moved j so that b_end_j <= a_start no longer holds.
            b_indices_for_this_a = []

            temp_j = j
            while temp_j < b_len:
                b_start, b_end = offsets_b[temp_j]

                # If we've gone past the end of the A token's range, stop.
                if b_start >= a_end:
                    break

                b_indices_for_this_a.append(temp_j)
                temp_j += 1

            # Store the B-indices that map to this A-token
            a_to_b_indices.extend(b_indices_for_this_a)
        a_to_b_indices = [idx for idx in a_to_b_indices if idx < max_length]
        if isinstance(mask_indices, torch.Tensor):
            return torch.Tensor(a_to_b_indices).type(mask_indices.dtype)
        return a_to_b_indices

    def _get_model_tokenizer(self, is_document_encoder):
        if is_document_encoder:
            return self.d_encoder, self.d_tokenizer
        else:
            return self.q_encoder, self.q_tokenizer

    def get_emb(
        self,
        texts=None,
        inputs=None,
        mask_indices=None,
        remove_indices=None,
        return_text=False,
        is_document_encoder=False,
        **kwargs,
    ):
        model, tokenizer = self._get_model_tokenizer(is_document_encoder)
        if "padding" not in kwargs:
            kwargs["padding"] = True
        if inputs is None:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                model.device
            )
        else:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        if mask_indices is not None:
            inputs["input_ids"][0][mask_indices] = tokenizer.mask_token_id

        if self.ret_type == "dpr":
            embeddings = model(**inputs).pooler_output
        else:
            outputs = model(**inputs)
            embeddings = self.mean_pooling(outputs[0], inputs["attention_mask"])
        if return_text:
            return embeddings, tokenizer.decode(inputs["input_ids"][0])
        return embeddings

    def get_mask_probs(self, text, mask_indices):
        using_mask_indices = (
            self.convert_tokens(text, mask_indices, self.reranker_tokenizer.model_max_length)
            if self.use_convert
            else mask_indices
        )
        original_token_probs = []
        for mask_idx in using_mask_indices:
            inputs = self.reranker_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
                self.d_encoder.device
            )
            original_idx = inputs["input_ids"][0][mask_idx].item()
            inputs["input_ids"][0][mask_idx] = self.reranker_tokenizer.mask_token_id
            logits = self.reranker(**inputs).logits.squeeze()
            probs = torch.softmax(logits, dim=-1)
            target = probs[mask_idx]
            original_token_probs.append(target[original_idx].item())
        return original_token_probs

    def get_attn(self, text1, text2=None, is_document_encoder=False, **kwargs):
        model, tokenizer = self._get_model_tokenizer(is_document_encoder)
        if text2 is None:
            inputs = tokenizer(text1, return_tensors="pt", padding=True, truncation=True).to(model.device)
        else:
            inputs = tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True).to(model.device)
        output = model(**inputs, output_attentions=True)
        return torch.stack(output["attentions"]).detach().cpu().squeeze()  # [layer, head, seq, seq]

    def get_last_hidden_states(self, text):
        model, tokenizer = self._get_model_tokenizer(is_document_encoder=True)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
