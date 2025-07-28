import json
import os
import time
from collections import defaultdict

from src.utils.gpt4 import create_batch_file, create_batch_item


def gpt4_measure(questions, docs, responses, prompt, save_path, client, additional_name=""):
    items = []
    for i in range(len(questions)):
        documents = ""
        for j, doc in enumerate(docs[i]):
            documents += f"[Document {j+1}] {doc.strip()}\n"
        cur_prompt = prompt.format_map(
            {"question": questions[i].strip(), "documents": documents, "response": responses[i].strip()}
        )
        item = create_batch_item(cur_prompt, idx=i, model="gpt-4o-2024-11-20")
        items.append(item)

    create_batch_file(client, save_path, items, "llm_eval", additional_name)


def gpt4_measure_by_answer(questions, correct_answers, responses, prompt, save_path, client, additional_name=""):
    items = []
    for i in range(len(questions)):
        correct_answer = correct_answers[i]
        try:
            cur_prompt = prompt.format_map(
                {"question": questions[i].strip(), "correct_answer": correct_answer, "response": responses[i].strip()}
            )
        except Exception as e:
            print(e)
            import pdb

            pdb.set_trace()
        item = create_batch_item(cur_prompt, idx=i, model="gpt-4o-2024-11-20")
        items.append(item)

    create_batch_file(client, save_path, items, "llm_eval", additional_name)


def load_batch_job(batch_path, client, batch_id=None):
    if not batch_id:
        file_path = os.path.join(batch_path)
        if not os.path.exists(file_path):
            raise ValueError(f"Batch id file not found for {batch_path}")

        with open(batch_path) as f:
            batch_id = f.read().strip()
    return client.batches.retrieve(batch_id)


def is_batch_ready(save_path, client, prefix_name, additional_names, batch_id=None, cur_try=0, max_try=36):
    for additional_name in additional_names:
        batch_path = os.path.join(save_path, f"batch_id-{prefix_name}{additional_name}.txt")
        batch_job = load_batch_job(batch_path, client, batch_id)
        if batch_job.status != "completed":
            print(f"[{cur_try+1}/{max_try}] Batch status : {batch_job.status}")
            return False
    return True


def get_responses_from_batch(save_path, client, prefix_name="", additional_names=[""], batch_id=None, max_try=180):
    if additional_names is None or additional_names == "":
        additional_names = [""]
    cur_try = 0
    while (
        not is_batch_ready(save_path, client, prefix_name, additional_names, batch_id, cur_try, max_try=max_try)
        and cur_try < max_try
    ):
        # time.sleep(300)
        time.sleep(60)
        cur_try += 1

    responses = defaultdict(list)
    for additional_name in additional_names:
        batch_path = os.path.join(save_path, f"batch_id-{prefix_name}{additional_name}.txt")
        batch_job = load_batch_job(batch_path, client, batch_id)
        result_file_id = batch_job.output_file_id
        result = client.files.content(result_file_id).content

        binary_path = os.path.join(save_path, f"batch_binary{additional_name}")
        with open(binary_path, "wb") as f:
            f.write(result)

        with open(binary_path, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                item = data["response"]["body"]["choices"][0]["message"]["content"]
                responses[additional_name[1:]].append(item)

    if additional_names == [""]:
        return responses[""]
    return responses


def print_gpt4_eval_results(save_path, client, batch_id=None, max_try=180, additional_names=["-clean", "-poison"]):
    final_output = ""
    answers = get_responses_from_batch(save_path, client, "llm_eval", additional_names, batch_id, max_try)
    for additional_name in additional_names:
        failed_output = ""
        total_cnt = 0
        total_correct = 0
        for i, item in enumerate(answers[additional_name[1:]]):
            found = False
            found_final_answer = False
            for chunk in item.split("\n"):
                if "final answer" in chunk.lower():
                    found_final_answer = True
                    if "yes" in chunk.lower():
                        total_correct += 1
                        found = True
                        break
                    elif "no" in chunk.lower():
                        found = True
                        break
                elif found_final_answer:
                    if "yes" in chunk.lower():
                        total_correct += 1
                        found = True
                        break
                    elif "no" in chunk.lower():
                        found = True
                        break

            if not found:
                failed_output += "====Fail====\n"
                failed_output += f"[{i}] {item}\n"
                failed_output += "==============\n"
            total_cnt += 1
        result_text = (
            f"[{additional_name[1:].upper()}] ({total_correct / total_cnt:.3f} ({total_correct}/{total_cnt}))\n"
        )
        final_output += result_text + failed_output
    return final_output, answers
