import argparse
import json
import os

import jsonlines
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Process dataset and save to output directory.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to load")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save processed files")
    parser.add_argument("--augmented", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset
    output_dir = args.output_dir

    collection = load_dataset(f"BeIR/{dataset}", "corpus")["corpus"]
    if args.augmented:
        q_path = os.path.join("data/poisoned_documents/phantom/augmented/contriever/llama2", f"{dataset}-200.json")
        q_data = json.load(open(q_path, "r", encoding="utf-8"))
        queries = [{"_id": item["query_id"], "text": item["query"]} for item in q_data]
    else:
        queries = load_dataset(f"BeIR/{dataset}", "queries")["queries"]

    os.makedirs(output_dir, exist_ok=True)

    collection_path = os.path.join(output_dir, "corpus.jsonl")
    query_path = os.path.join(output_dir, "queries.tsv")

    with jsonlines.open(collection_path, mode="w") as writer:
        for row in collection:
            line = {"id": row["_id"], "contents": row["title"] + "\n" + row["text"].replace("\n", " ")}
            writer.write(line)

    with open(query_path, mode="w") as writer:
        for row in queries:
            writer.write(f"{row['_id']}\t{row['text']}\n")


if __name__ == "__main__":
    main()
