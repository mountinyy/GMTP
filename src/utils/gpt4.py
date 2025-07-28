import json
import os
from datetime import datetime


def create_batch_item(prompt, idx=0, model="gpt-4o-2024-11-20", system_prompt="You are a helpful assistant."):
    item = {
        "custom_id": f"result-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        },
    }
    return item


def create_batch_file(client, save_path, items, prefix_name, additional_name=""):
    os.makedirs(save_path, exist_ok=True)
    cur_time = datetime.now().strftime("%m%d%H%M")
    file_path = os.path.join(save_path, f"batch_file-{prefix_name}{additional_name}-{cur_time}.jsonl")
    print("Save batch file in ", file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    batch_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch",
    )
    batch_job = client.batches.create(
        input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h"
    )
    print("Batch job id : ", batch_job.id)
    batch_id_path = os.path.join(save_path, f"batch_id-{prefix_name}{additional_name}.txt")
    with open(batch_id_path, "w", encoding="utf-8") as f:
        f.write(batch_job.id)
    print("Batch id saved at ", batch_id_path)
