import argparse
import os

import faiss

parser = argparse.ArgumentParser()
parser.add_argument("--dimension", type=int, help="dimension of passage embeddings", required=False, default=768)
parser.add_argument("--index_dir", type=str, help="directory to store brute force index of corpus", required=True)
parser.add_argument(
    "--atk_index_dir", type=str, help="directory to store brute force poisoned index of corpus", required=True
)
parser.add_argument(
    "--full_index_dir", type=str, help="directory to store brute force full index of corpus", required=True
)
args = parser.parse_args()

new_index = faiss.IndexFlatIP(args.dimension)
docid_files = []

# add normal index
index = faiss.read_index(os.path.join(args.index_dir, "index"))
docid_files.append(os.path.join(args.index_dir, "docid"))
vectors = index.reconstruct_n(0, index.ntotal)
new_index.add(vectors)

# add attack index
index = faiss.read_index(os.path.join(args.atk_index_dir, "index"))
docid_files.append(os.path.join(args.atk_index_dir, "docid"))
vectors = index.reconstruct_n(0, index.ntotal)
new_index.add(vectors)

if not os.path.exists(args.full_index_dir):
    os.makedirs(args.full_index_dir, exist_ok=True)
faiss.write_index(new_index, os.path.join(args.full_index_dir, "index"))

with open(os.path.join(args.full_index_dir, "docid"), "w") as wfd:
    for f in docid_files:
        with open(f, "r") as f1:
            for line in f1:
                wfd.write(line)
