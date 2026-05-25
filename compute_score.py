import os
import argparse
import pickle
import torch
import tqdm
from transformers import AutoTokenizer
from build_prefix_trie import VocabTrieNode

def get_prefix_masks(
    batch_token_ids, 
    tokenizer, 
    prefix_trie
):
    """
    Use a trie-based lookup to obtain prefix masks.
    """
    prefix_strs = tokenizer.batch_decode(batch_token_ids, skip_special_tokens=True)
    prefix_strs = [p.strip().lower() for p in prefix_strs]

    batch_size = len(prefix_strs)
    vocab_size = len(tokenizer)

    mask_tensor = torch.zeros(
        (batch_size, vocab_size),
        dtype=torch.bool
    )

    for i, prefix in enumerate(prefix_strs):
        tids = []
        node = prefix_trie
        for ch in prefix:
            tids.extend(node.end_token_ids)
            if ch not in node.children:
                break
            node = node.children[ch]
        else:
            tids.extend(node.end_token_ids)

        if tids:
            mask_tensor[i, tids] = True

    return mask_tensor

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.huggingface_token)
        
    with open(f"{args.output_dir}/{args.dataset}/{args.model}/generation.pkl", "rb") as f:
        generation = pickle.load(f)

    with open(f"{args.output_dir}/{args.dataset}/{args.model}/log_prob.pkl", "rb") as f:
        log_prob = pickle.load(f)

    with open(f"{args.precompute_dir}/{args.model}/embedding_matrix.pkl", "rb") as f:
        embedding_matrix = pickle.load(f)

    with open(f"{args.precompute_dir}/{args.model}/prefix_trie.pkl", "rb") as f:
        prefix_trie = pickle.load(f)
    
    log_score = []

    for g in range(len(generation[0]["output"])):

        log_prob_g = [_[g] for _ in log_prob]

        # Flatten log probs in one call
        log_prob_flat = torch.cat(log_prob_g, dim=0)          # (total_tokens, vocab)

        generated_token_flat = []
        subsequent_token_flat = []
        for _i in range(len(log_prob_g)):
            tokens = generation[_i]["output_token"][g]
            generated_token_flat.extend(tokens)
            subsequent_token_flat.extend(tokens[_j:] for _j in range(len(tokens)))
        generated_token_flat = torch.tensor(generated_token_flat)

        
        clustered_log_prob_flat = []
        for st in tqdm.tqdm(range(0, len(log_prob_flat), args.batch_size)):
            ed = min(st + args.batch_size, len(log_prob_flat))

            log_prob_batch  = log_prob_flat[st:ed]       # (B,vocab)
            idx_batch   = generated_token_flat[st:ed]    # (B,)
            mask_embedding = embedding_matrix[idx_batch] # (B,vocab)

            mask_prefix = get_prefix_masks(subsequent_token_flat[st:ed], tokenizer, prefix_trie)           # (B,vocab)
            mask_unified = mask_embedding | mask_prefix
            log_prob_batch_masked = log_prob_batch.masked_fill(~mask_unified, -float("inf"))

            # log of the cluster probability mass
            log_prob_batch_masked_sum = torch.logsumexp(
                        log_prob_batch_masked,
                        dim=-1)

            clustered_log_prob_flat.append(log_prob_batch_masked_sum)
        clustered_log_prob_flat = torch.concat(clustered_log_prob_flat, dim=0)


        # Score each sample
        sample_lengths = [len(sample["output_token"][g]) for sample in generation]
        splits = torch.split(clustered_log_prob_flat, sample_lengths)

        log_score_g = torch.full((len(generation),), -float("inf"))
        for i, sample in enumerate(generation):
            if sample["output"][g].strip() != "":
                log_score_g[i] = splits[i].sum()
        log_score.append(log_score_g)
    log_score = torch.stack(log_score, 0)

    save_path = f"{args.save_dir}/{args.dataset}/{args.model}/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path+"log_score.pkl", "wb") as f:
        pickle.dump(log_score, f)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="trivia_qa")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--precompute_dir", type=str, default="./precompute")
    parser.add_argument("--save_dir", type=str, default="./score")
    parser.add_argument("--batch_size", type=int, default=2000000)
    parser.add_argument("--huggingface_token", type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
