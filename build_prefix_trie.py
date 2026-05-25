import os
import argparse
import pickle
from transformers import AutoTokenizer


class VocabTrieNode:
    __slots__ = ["children", "end_token_ids"]
    def __init__(self):
        # children[ch] => child node for character ch
        self.children = {}
        # end_token_ids => list of token IDs that end exactly at this node
        self.end_token_ids = []


def build_prefix_trie(tokenizer):
    """
    Build and return a trie structure for all vocabulary entries
    with the format: stripped & lowercased text -> token ID(s).
    """
    root = VocabTrieNode()

    for token_id in range(len(tokenizer)):
        token_str = tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
        if token_str == "":
            continue

        node = root
        for ch in token_str:
            if ch not in node.children:
                node.children[ch] = VocabTrieNode()
            node = node.children[ch]
        # Store that a token ends at this node
        node.end_token_ids.append(token_id)

    return root


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", token=args.huggingface_token)
    
    prefix_trie = build_prefix_trie(tokenizer)

    save_path = f"{args.save_dir}/{args.model}/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path+"prefix_trie.pkl", "wb") as f:
        pickle.dump(prefix_trie, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--save_dir", type=str, default="./precompute")
    parser.add_argument("--huggingface_token", type=str, default=None)
    args = parser.parse_args()
    main(args)
    