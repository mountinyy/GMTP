from src.utils.beir_utils import load_dataset as load_beir_dataset

datasets = ["nq", "hotpotqa", "msmarco"]

for dataset in datasets:
    corpus, queries, qrels = load_beir_dataset(dataset, "test" if dataset == "nq" else "train")
