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


class DPR:
    def __init__(self, model_name, use_cuda=True, is_question_encoder=False, reranker="bert-base-uncased"):
        model_class = DPRQuestionEncoder if is_question_encoder else DPRContextEncoder
        tokenizer_class = DPRQuestionEncoderTokenizerFast if is_question_encoder else DPRContextEncoderTokenizerFast
        self.model = model_class.from_pretrained(model_name)
        if use_cuda:
            self.model.to("cuda")
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.tokenizer.model_max_length = 512
        if not is_question_encoder:
            self.reranker = AutoModelForMaskedLM.from_pretrained(reranker).eval().to(self.model.device)
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker)
            self.use_convert = self.tokenizer.get_vocab() != self.reranker_tokenizer.get_vocab()
            if self.use_convert:
                print("Using different vocabularies for DPR and Reranker tokenizers.", flush=True)

    def convert_tokens(self, text, mask_indices, max_length=512):
        # Tokenize the text with both tokenizers, asking for offset mappings
        encoding_a = self.tokenizer(text, return_offsets_mapping=True)
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

    def get_emb(self, texts=None, inputs=None, mask_indices=None, remove_indices=None, return_text=False, **kwargs):
        if "padding" not in kwargs:
            kwargs["padding"] = True
        if inputs is None:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.model.device
            )
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        if mask_indices is not None:
            inputs["input_ids"][0][mask_indices] = self.tokenizer.mask_token_id
        elif remove_indices is not None:
            mask = torch.ones(inputs["input_ids"].size(-1), dtype=torch.bool)
            mask[remove_indices] = False
            for k in inputs:
                target_tensor = inputs[k][0][mask].clone()
                inputs[k] = inputs[k][:, : mask.sum()]
                inputs[k][0] = target_tensor

        embeddings = self.model(**inputs).pooler_output
        if return_text:
            return embeddings, self.tokenizer.decode(inputs["input_ids"][0])
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
                self.model.device
            )
            original_idx = inputs["input_ids"][0][mask_idx].item()
            inputs["input_ids"][0][mask_idx] = self.reranker_tokenizer.mask_token_id
            logits = self.reranker(**inputs).logits.squeeze()
            probs = torch.softmax(logits, dim=-1)
            target = probs[mask_idx]
            original_token_probs.append(target[original_idx].item())
        return original_token_probs

    def get_attn(self, text1, text2=None, **kwargs):
        if text2 is None:
            inputs = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        else:
            inputs = self.tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True).to(
                self.model.device
            )
        output = self.model(**inputs, output_attentions=True)
        return torch.stack(output["attentions"]).detach().cpu().squeeze()  # [layer, head, seq, seq]

    def get_last_hidden_states(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state


class Contriever:
    def __init__(self, model_name, use_cuda=True, is_question_encoder=False, reranker="bert-base-uncased"):
        self.model = AutoModel.from_pretrained(model_name)
        if use_cuda:
            self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = 512
        if not is_question_encoder:
            self.reranker = AutoModelForMaskedLM.from_pretrained(reranker).eval().to(self.model.device)
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker)
            self.use_convert = self.tokenizer.get_vocab() != self.reranker_tokenizer.get_vocab()
            if self.use_convert:
                print("Using different vocabularies for Contriever and Reranker tokenizers.", flush=True)

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def convert_tokens(self, text, mask_indices, max_length=512):
        # Tokenize the text with both tokenizers, asking for offset mappings
        encoding_a = self.tokenizer(text, return_offsets_mapping=True)
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

    def get_mask_probs(self, text, mask_indices):
        using_mask_indices = (
            self.convert_tokens(text, mask_indices, self.reranker_tokenizer.model_max_length)
            if self.use_convert
            else mask_indices
        )
        original_token_probs = []
        for mask_idx in using_mask_indices:
            inputs = self.reranker_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
                self.model.device
            )
            original_idx = inputs["input_ids"][0][mask_idx].item()
            inputs["input_ids"][0][mask_idx] = self.reranker_tokenizer.mask_token_id
            logits = self.reranker(**inputs).logits.squeeze()
            probs = torch.softmax(logits, dim=-1)
            target = probs[mask_idx]
            original_token_probs.append(target[original_idx].item())
        return original_token_probs

    def get_emb(self, texts=None, inputs=None, mask_indices=None, return_text=False, **kwargs):
        if "padding" not in kwargs:
            kwargs["padding"] = True
        if inputs is None:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.model.device
            )
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        if mask_indices is not None:
            inputs["input_ids"][0][mask_indices] = self.tokenizer.mask_token_id
        outputs = self.model(**inputs)
        embeddings = self.mean_pooling(outputs[0], inputs["attention_mask"])
        if return_text:
            return embeddings, self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        return embeddings

    def get_attn(self, text1, text2=None, **kwargs):
        if text2 is None:
            inputs = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        else:
            inputs = self.tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True).to(
                self.model.device
            )
        output = self.model(**inputs, output_attentions=True)
        return torch.stack(output["attentions"]).detach().cpu().squeeze()  # [layer, head, seq, seq]

    def get_last_hidden_states(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
