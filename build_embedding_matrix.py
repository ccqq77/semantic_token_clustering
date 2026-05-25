from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import pickle
import gc
from sklearn.cluster import AgglomerativeClustering
import nltk
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
import numpy as np
import os


def precompute_cosine_distance(embedding):
    """
    Computes pairwise cosine distances (1 - cosine_similarity).
    Equivalent to sklearn.metrics.pairwise.cosine_distances(embedding)
    """
    embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=1)
    cosine_dist = 1 - torch.mm(embedding_norm, embedding_norm.t())
    cosine_dist = torch.clamp(cosine_dist, min=0.0)
    return cosine_dist


def precompute_euclidean_distance(embedding):
    """
    Computes pairwise Euclidean distances.
    Uses the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
    Equivalent to sklearn.metrics.pairwise.euclidean_distances(embedding)
    """
    # Compute squared norms for each row
    sq_norms = (embedding ** 2).sum(dim=1, keepdim=True)  # (N, 1)

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x·y
    dist_sq = sq_norms + sq_norms.t() - 2.0 * torch.mm(embedding, embedding.t())

    # Clamp to avoid negative values due to floating point errors
    dist_sq = torch.clamp(dist_sq, min=0.0)

    return torch.sqrt(dist_sq)


def contains_number(s: str) -> bool:
    return any(char.isdigit() for char in s)

def main(args):

    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", token=args.huggingface_token)

    if args.embedding == "input":
        embedding = model.get_input_embeddings().weight.data[:len(tokenizer), :].clone()
    elif args.embedding == "output":
        embedding = model.get_output_embeddings().weight.data[:len(tokenizer), :].clone()
    elif args.embedding == "concatenated":
        embedding_in = model.get_input_embeddings().weight.data[:len(tokenizer), :].clone()
        embedding_in = embedding_in / torch.clamp(torch.norm(embedding_in, p=2, dim=1, keepdim=True), min=1e-9)
        embedding_out = model.get_output_embeddings().weight.data[:len(tokenizer), :].clone()
        embedding_out = embedding_out / torch.clamp(torch.norm(embedding_out, p=2, dim=1, keepdim=True), min=1e-9)
        embedding = torch.cat((embedding_in, embedding_out), dim=-1)
    else:
        raise ValueError(f"Undefined embedding type: {args.embedding}. Choose from 'input', 'output', 'concatenated'.")

    del model
    gc.collect()

    if args.precompute_distance:
        if args.metric == "cosine":
            distance = precompute_cosine_distance(embedding).float().numpy(force=True)
        elif args.metric == "euclidean":
            distance = precompute_euclidean_distance(embedding).float().numpy(force=True)
        else:
            raise ValueError(f"Undefined metric: {args.metric}. Choose from 'cosine', 'euclidean'.")
        
        del embedding
        gc.collect()
        results = AgglomerativeClustering(n_clusters=args.n_clusters, metric="precomputed", linkage=args.linkage).fit(distance)
        del distance
        gc.collect()
        
    else:
        embedding = embedding.float().numpy(force=True)
        results = AgglomerativeClustering(n_clusters=args.n_clusters, metric=args.metric, linkage=args.linkage).fit(embedding)
        del embedding
        gc.collect()

    cluster_labels = torch.tensor(results.labels_)

    exclude_list = []
    for _i in range(len(tokenizer)):
        if contains_number(tokenizer.decode(_i, skip_special_tokens=True)):
            exclude_list.append(_i)
    exclude_tensor = torch.as_tensor(exclude_list, dtype=torch.long)
    
    stopwords_list = []
    decoded_vocab = tokenizer.batch_decode(range(len(tokenizer)), skip_special_tokens=True)
    for _i, _tok in enumerate(decoded_vocab):
        if _tok.strip() in stopwords.words("english") or _tok.strip() == "":
            stopwords_list.append(_i)
    stopwords_tensor = torch.as_tensor(stopwords_list, dtype=torch.long)

    embedding_matrix = (
        cluster_labels.unsqueeze(0) == cluster_labels.unsqueeze(1)
    )                                                    # (V, V) same-cluster
    embedding_matrix[exclude_tensor] = False
    embedding_matrix[exclude_tensor, exclude_tensor] = True
    embedding_matrix[stopwords_tensor] = True
    embedding_matrix[:, stopwords_tensor] = True
    
    save_path = f"{args.save_dir}/{args.model}/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path+"embedding_matrix.pkl", "wb") as f:
        pickle.dump(embedding_matrix, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--embedding", type=str, default="concatenated")
    parser.add_argument("--n_clusters", type=int, default=16000)
    parser.add_argument("--linkage", type=str, default="complete")
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--precompute_distance", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./precompute")
    parser.add_argument("--huggingface_token", type=str, default=None)

    args = parser.parse_args()
    main(args)
