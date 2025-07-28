# from beir import util
from beir import util
from beir.datasets.data_loader import GenericDataLoader


def load_dataset(dataset, split):
    """load beir datasets

    Args:
        dataset (str): name of datasets. nq, hotpotqa, msmarco
        split (str): train, dev, test

    Returns:
        corpus (dict): dictionary of corpus
        queries (dict): dictionary of queries
        qrels (dict): dictionary of qrels
    """
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    if dataset == "nq" and split == "train":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq-train.zip"
    data_path = util.download_and_unzip(url, "data/beir_github")

    return GenericDataLoader(data_folder=data_path).load(split=split)
