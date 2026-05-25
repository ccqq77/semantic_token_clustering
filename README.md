# Semantic Token Clustering (STC)

Code for the EACL 2026 paper
[**Semantic Token Clustering for Efficient Uncertainty Quantification in Large Language Models**](https://aclanthology.org/2026.eacl-short.49/)
([arXiv:2603.20161](https://arxiv.org/abs/2603.20161)).

STC is a single-pass, auxiliary-model-free uncertainty quantification method for LLMs. At each decoding step, it aggregates probability mass over a semantic cluster of vocabulary tokens, defined by (i) embedding-based clustering of the token vocabulary and (ii) prefix matching that identifies alternative tokenizations of the same answer continuation.

## Requirements

The direct dependencies are listed in `requirements.txt`. The code has been tested with Python 3.12, PyTorch 2.7, Transformers 4.52, and FlashAttention 2 on CUDA GPUs.


## Pipeline

The pipeline has four steps. All scripts share `--model` (HuggingFace ID). Outputs are written under `./output`, `./precompute`, and `./score` by default.

Steps 1 and 2 produce model-dependent artifacts that are precomputed once per model and reused across all datasets. Step 3 generates answers and log-probabilities on a given dataset, and step 4 combines everything into the STC score.

For gated HuggingFace models (e.g. Llama), pass an access token via `--huggingface_token` to any of the four scripts, or run `huggingface-cli login` once beforehand.

### 1. Build the vocabulary clustering matrix

```bash
python build_embedding_matrix.py \
    --model meta-llama/Llama-2-7b-hf \
    --embedding concatenated \
    --n_clusters 16000 \
    --linkage complete \
    --metric cosine \
    --precompute_distance \
    --save_dir ./precompute
```

This runs agglomerative clustering on token embeddings (input, output, or concatenation) and saves a boolean `(V, V)` matrix indicating which vocabulary tokens belong to the same cluster. Numeric tokens are excluded and stop-word rows/columns are treated as universal matches.

### 2. Build the prefix trie

```bash
python build_prefix_trie.py \
    --model meta-llama/Llama-2-7b-hf \
    --save_dir ./precompute
```

Builds a character-level trie over the (stripped, lowercased) vocabulary, used to identify all vocabulary tokens whose surface form is a prefix of a continuation.

### 3. Generate answers and log-probabilities

```bash
python generate.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset trivia_qa \
    --batch_size 64 \
    --num_return_sequences 1 \
    --temperature 0.0 \
    --top_k 0 \
    --top_p 1.0 \
    --log_prob \
    --max_length 256 \
    --save_dir ./output
```

Supported datasets: `nq`, `trivia_qa`, `web_questions`. The flag `--log_prob` is required for the STC score; `--logit` and `--hidden_state` are optional dumps.

### 4. Compute STC scores

```bash
python compute_score.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset trivia_qa \
    --output_dir ./output \
    --precompute_dir ./precompute \
    --save_dir ./score
```

At each step, this aggregates the model's probability mass over the union of the embedding-cluster and prefix-match tokens, and combines the per-step values into a per-sample log-score.

## Output layout

```
output/<dataset>/<model>/{generation,log_prob}.pkl
precompute/<model>/{embedding_matrix,prefix_trie}.pkl
score/<dataset>/<model>/log_score.pkl
```

## Citation

```bibtex
@inproceedings{cao-etal-2026-semantic,
    title = "Semantic Token Clustering for Efficient Uncertainty Quantification in Large Language Models",
    author = "Cao, Qi  and
      Gambardella, Andrew  and
      Kojima, Takeshi  and
      Matsuo, Yutaka  and
      Iwasawa, Yusuke",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 2: Short Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-short.49/",
    doi = "10.18653/v1/2026.eacl-short.49",
    pages = "682--696",
    ISBN = "979-8-89176-381-4"
}
```
