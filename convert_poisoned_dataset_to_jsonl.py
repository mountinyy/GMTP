import argparse
import json
import os

import jsonlines


def main():
    parser = argparse.ArgumentParser(description="Process dataset and save to output directory.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to load")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save processed files")
    args = parser.parse_args()

    print("Loading dataset from", args.dataset)
    print("Converting dataset into ", args.output_dir)
    custom_data = True if "augmented" in args.dataset else False
    if custom_data:
        print("Using Custom dataset")

    output_dir = args.output_dir

    with open(args.dataset, encoding="utf-8") as f:
        poisoned_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    collection_path = os.path.join(output_dir, "corpus.jsonl")
    query_path = os.path.join(output_dir, "queries.tsv")

    poisoned_docs = set()
    for data in poisoned_data:
        for poisoned_doc in data["poisoned_docs"]:
            poisoned_docs.add(poisoned_doc)

    with jsonlines.open(collection_path, mode="w") as writer:
        id = 0
        for poisoned_doc in poisoned_docs:
            line = {"id": f"poisoned_{id}", "contents": poisoned_doc}
            writer.write(line)
            id += 1

    with open(query_path, mode="w") as writer:
        for i, data in enumerate(poisoned_data):
            writer.write(f"{data['query_id']}\t{data['query']}\n")


if __name__ == "__main__":
    main()
