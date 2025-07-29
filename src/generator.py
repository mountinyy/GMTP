import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = """"You are a helpful assistant.
Your task is to answer user queries accurately and concisely based on the information provided in the given contexts.
You may generalize or infer an answer if it is logically supported by the context, but do not introduce information that is not present or cannot be reasonably inferred.
If the answer can be found directly or inferred from the context, provide a clear and concise response, without additional information. 
If the context does not contain enough information to answer the question, respond with 'I don't know.'
Ensure your response only contains the answer to the query, without additional explanation.

Contexts: [context]

Query: [question]

Answer:"""

PRIORITY_PROMPT = """You are a highly intelligent and helpful assistant.
Your task is to answer user queries accurately and concisely based on the provided contexts, which may include a priority score indicating their reliability or importance.
When multiple contexts contain conflicting information, prioritize the information from the context with the higher priority score.
If no priority score is given or if contexts have the same score, use the context that best aligns with the query.
You may generalize or infer an answer if it is logically supported by the provided contexts but do not introduce information that is not present or cannot be reasonably inferred.
If the answer can be found directly or inferred from the context, provide a clear and concise response, without additional information. 
If the contexts do not contain enough information to answer the question, respond with 'I don't know.'
Ensure your response only contains the answer to the query.


#1.
Contexts: [Priority 1] John is 13 years old.
[Priority 2] John is 14 years old.

Query: How old is John?

Answer: 14 years old.

#2.
Contexts: [context]

Query: [question]

Answer:"""


class Generator:
    def __init__(self, conf):
        if conf.model.generator == "llama3":
            model_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif conf.model.generator == "llama2":
            model_name = "meta-llama/Llama-2-7b-chat-hf"
        elif conf.model.generator == "gemma2":
            model_name = "google/gemma-2-2b-it"
        elif conf.model.generator == "qwen":
            model_name = "Qwen/Qwen2.5-7B-Instruct"
        elif conf.model.generator == "mistral7b":
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.temperature = conf.rag.temperature
        self.max_new_tokens = conf.rag.max_new_tokens
        self.max_output_tokens = self.max_new_tokens

    def generate(self, question, docs, use_priority=False, is_gt=False):
        prompt = wrap_prompt(question, docs, use_priority)
        tok = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        outputs = self.model.generate(
            **tok, temperature=self.temperature, max_new_tokens=self.max_new_tokens, early_stopping=True
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output.split("Answer:")[-1].strip()

        return response

    def __call__(self, inputs, params):
        if isinstance(inputs, list):
            outputs = []
            for p in tqdm(inputs, total=len(inputs), desc="Generating responses"):
                input_ids = self.tokenizer.encode(p, return_tensors="pt").to(self.model.device)
                output = self.model.generate(input_ids, do_sample=True, **params)
                output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                outputs.append(output[len(p) :].strip())
            return outputs
        input_ids = self.tokenizer.encode(inputs, return_tensors="pt").to(self.model.device)
        output = self.model.generate(input_ids, do_sample=True, **params)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output[len(p) :].strip()

    def get_prob(
        self,
        question=None,
        docs=None,
        adv_g=None,
        target_response=None,
        tok=None,
        labels=None,
        use_priority=False,
        device="cpu",
    ):
        if tok is None and labels is None:
            prompt = wrap_prompt(question, docs, use_priority)
            prompt += target_response
            tok = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            labels = torch.full_like(tok["input_ids"], -100)

            target_idx = self.tokenizer(
                target_response, return_tensors="pt", truncation=True, add_special_tokens=False
            )["input_ids"][0]
            adv_g_idx = self.tokenizer(adv_g, return_tensors="pt", truncation=True, add_special_tokens=False)[
                "input_ids"
            ][0]
            found = False
            for i in range(tok["input_ids"].size(-1)):
                if (tok["input_ids"][0, i : i + len(target_idx)] == target_idx).sum() > int(len(target_idx) * 0.8):
                    labels[0, i : i + len(target_idx)] = target_idx
                    found = True
                    break

            if not found:
                raise ValueError("Target response not found in prompt.")
            for i in range(tok["input_ids"].size(-1)):
                if (tok["input_ids"][0, i : i + len(adv_g_idx)] == adv_g_idx).sum() > int(len(adv_g_idx) * 0.8):
                    adv_g_start = i
                    adv_g_end = i + len(adv_g_idx)
                    break
        else:
            adv_g_start, adv_g_end = None, None
        tok = {k: v.to(device) for k, v in tok.items()}
        labels = labels.to(device)
        outputs = self.model(**tok, labels=labels, use_cache=True)
        return outputs.loss, (tok, labels), (adv_g_start, adv_g_end)


def wrap_prompt(question, contexts, use_priority=False):
    prompt = PRIORITY_PROMPT if use_priority else DEFAULT_PROMPT
    context_str = ""
    prefix = "Priority" if use_priority else "Document"
    contexts = contexts[::-1] if use_priority else contexts
    for i in range(len(contexts)):
        context_str += f"[{prefix} {i+1}] {contexts[i]}\n"
    input_prompt = prompt.replace("[question]", question).replace("[context]", context_str)

    return input_prompt
