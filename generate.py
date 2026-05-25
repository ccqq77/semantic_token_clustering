# Parts of this script are adapted from 
# https://github.com/zlin7/UQ-NLG/

import torch
import torch.nn.functional as F
import numpy as np
import datasets
import argparse
import os
import functools
import pickle
import unicodedata
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, GenerationConfig
from tqdm import tqdm


def find_newline_token_ids(tokenizer):
    newline_token = tokenizer.convert_ids_to_tokens(tokenizer.encode("\n"))[-1]
    newline_token_ids = []
    for token, token_id in tokenizer.get_vocab().items():
        if newline_token in token:
            newline_token_ids.append(token_id)
    return newline_token_ids


def _valid_len(seqs: torch.Tensor, eos_ids: torch.Tensor) -> torch.Tensor:
    """
    seqs   : (N , L)  on CPU – generated tokens *after* the prompt
    eos_ids: (n_eos,) on CPU – every EOS id
    returns: (N,)     on CPU – usable length for each sequence
    """
    eos_mask      = torch.isin(seqs, eos_ids)
    first_eos_pos = eos_mask.float().argmax(dim=1)
    has_eos       = eos_mask.any(dim=1)
    full_len      = torch.full_like(first_eos_pos, seqs.size(1))
    return torch.where(has_eos, first_eos_pos + 1, full_len)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_arguments()

    set_seed(args.random_seed)

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="auto", attn_implementation="flash_attention_2", token=args.huggingface_token)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", token=args.huggingface_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    additional_eos = find_newline_token_ids(tokenizer)
    all_eos = list(set(additional_eos + [tokenizer.eos_token_id]))
    all_eos_cpu = torch.tensor(all_eos)

    if args.dataset == "nq":
        @functools.lru_cache()
        def get_fs_samples_prompt():
            data = datasets.load_dataset("google-research-datasets/nq_open", split="train")
            indices = np.random.RandomState(42).choice(len(data), 5)
            ret = ""
            for idx in indices:
                idx = int(idx)
                ret += f"Question:\n{data[idx]['question']}\nAnswer:\n{data[idx]['answer'][0]}\n\n"
            return ret

        def sample_to_prompt(sample, **kwargs):
            if isinstance(sample["question"], list):
                return [sample_to_prompt({"question": _}, **kwargs) for _ in sample["question"]]
            return f"""Answer these questions:\n\n{get_fs_samples_prompt()}Question:\n{sample['question']}\nAnswer:\n"""

        dataset = datasets.load_dataset("google-research-datasets/nq_open", split="validation")
        input_prompt = [sample_to_prompt(_) for _ in dataset]

    elif args.dataset == "trivia_qa":
        data = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()
        def remove_dups(batch):
            if batch["question_id"][0] in id_mem:
                return {_:[] for _ in batch.keys()}
            id_mem.add(batch["question_id"][0])
            return batch
        dataset = data.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
        def sample_to_prompt(sample, **kwargs):
            return f"""Answer these questions:\n\nQuestion:\nIn Scotland a bothy/bothie is a?\nAnswer:\nHouse\n\nQuestion:\n{sample['question']}\nAnswer:\n"""

        input_prompt = [sample_to_prompt(_) for _ in dataset]

    elif args.dataset == "web_questions":
        def rename_feature(batch):
            batch["answer"] = batch.pop("answers")
            return batch
        
        dataset = datasets.load_dataset("stanfordnlp/web_questions", split="test", trust_remote_code=True)
        dataset = dataset.map(rename_feature, batched=True, remove_columns=["answers"])
        def sample_to_prompt(sample, **kwargs):
            return f"""Answer these questions:\n\nQuestion:\nwhere was the ancient region of mesopotamia?\nAnswer:\nMiddle East\n\nQuestion:\n{sample['question']}\nAnswer:\n"""

        input_prompt = [sample_to_prompt(_) for _ in dataset]

    else:
        raise ValueError(f"Undefined dataset: {args.dataset}. Choose from 'nq', 'trivia_qa', 'web_questions'.")

    input_prompt = [unicodedata.normalize("NFKD", _) for _ in input_prompt]


    logit = []
    log_prob = []
    generation = []
    state = []
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    for i in tqdm(range(0, len(input_prompt), args.batch_size)):
        x = input_prompt[i:i+args.batch_size]
        output_line = {}
        output_line["input"] = x
        x_token = tokenizer(x, return_tensors="pt", padding=True).to("cuda")


        if args.temperature == 0:
            n_ret = 1
            generation_config = {
                "max_new_tokens":args.max_length,
                "do_sample":False,
                "bos_token_id":tokenizer.bos_token_id,
                "eos_token_id":all_eos,
                "pad_token_id":tokenizer.pad_token_id,
                "return_dict_in_generate":True,
                "output_logits":args.logit or args.log_prob,
                "output_hidden_states":args.hidden_state,
                "num_beams":1,
            }
        else:
            n_ret = args.num_return_sequences
            generation_config = {
                "max_new_tokens":args.max_length,
                "do_sample":True,
                "temperature":args.temperature,
                "num_return_sequences":n_ret,
                "bos_token_id":tokenizer.bos_token_id,
                "eos_token_id":all_eos,
                "pad_token_id":tokenizer.pad_token_id,
                "return_dict_in_generate":True,
                "output_logits":args.logit or args.log_prob,
                "output_hidden_states":args.hidden_state,
                "top_p":args.top_p,
                "num_beams":1,
                "top_k":args.top_k,
            }
        generation_config = GenerationConfig(**generation_config)

        with torch.no_grad():
            output = model.generate(
                inputs=x_token.input_ids,
                attention_mask=x_token.attention_mask,
                generation_config=generation_config
                )

            
        # ───────────────────────────── shortcuts ────────────────────────────────────
        prompt_len = x_token.input_ids.size(1)              # prompt length  (int)
        bs         = x_token.input_ids.size(0)              # batch size

        # ───────────────────────── 1) tokens (CPU) ──────────────────────────────────
        seqs_cpu   = output.sequences.to("cpu").detach()    # (bs*n_ret , prompt+gen)
        seqs_after = seqs_cpu[:, prompt_len:]               # drop prompt
        L_gen      = seqs_after.size(1)


        # usable length for every sequence (vectorised, CPU)
        valid_len_flat = _valid_len(seqs_after, all_eos_cpu)           # (bs*n_ret,)
        valid_len      = valid_len_flat.view(bs, n_ret)                # (bs , n_ret)

        # ───────────────────────── 2) decode to text (CPU) ─────────────────────────
        decoded_flat = tokenizer.batch_decode(seqs_after, skip_special_tokens=True)
        decoded_txt  = [decoded_flat[j:j+n_ret] for j in range(0, len(decoded_flat), n_ret)]

        # ───────────────────────── 3) token-lists + lengths ────────────────────────
        seqs_after = seqs_after.view(bs, n_ret, L_gen)                 # (bs , n_ret , L_gen)

        lst_token_id, lst_num_valid = [], []
        for b in range(bs):
            tid_row, len_row = [], []
            for r in range(n_ret):
                v = valid_len[b, r].item()
                len_row.append(v)
                tid_row.append(seqs_after[b, r, :v].tolist())
            lst_token_id.append(tid_row)
            lst_num_valid.append(len_row)

        output_line["output"]       = decoded_txt
        output_line["output_token"] = lst_token_id

        # ───────────────────────── 4) logit / log_prob (CPU) ──────────────────────
        if args.logit or args.log_prob:
            # bring logits to CPU *once*
            logit_cpu = torch.stack(
                [step.to("cpu").detach() for step in output.logits], dim=0
            )                                              # (L_gen , bs*n_ret , vocab)
            logit_cpu = logit_cpu.permute(1, 0, 2)         # (bs*n_ret , L_gen , vocab)
            logit_cpu = logit_cpu.view(bs, n_ret, L_gen, -1)

            if args.log_prob:
                log_prob_cpu = F.log_softmax(logit_cpu[..., :len(tokenizer)], dim=-1)

            for b in range(bs):
                logit_row, log_prob_row = [], []
                for r in range(n_ret):
                    v = valid_len[b, r].item()
                    if args.logit:
                        logit_row.append(logit_cpu[b, r, :v].clone())
                    if args.log_prob:
                        log_prob_row.append(log_prob_cpu[b, r, :v].clone())
                if args.logit:
                    logit.append(logit_row)
                if args.log_prob:
                    log_prob.append(log_prob_row)

        # ───────────────────────── 5) hidden state (CPU) ──────────────────────────
        if args.hidden_state:
            # Collect the hidden-state of the newly generated token at each step,
            # moving each slice to CPU immediately.
            state_per_step = []
            for step in output.hidden_states:  # step: tuple(num_layers) – GPU tensors
                last_tok = [layer[:, -1, :].to("cpu").detach() for layer in step]
                state_per_step.append(torch.stack(last_tok, dim=1))      # (bs*n_ret , num_layers , hidden)

            state_cpu = torch.stack(state_per_step, dim=1)              # (bs*n_ret , L_gen , num_layers , hidden)
            state_cpu = state_cpu.view(bs, n_ret,
                                        state_cpu.size(1),
                                        state_cpu.size(2),
                                        state_cpu.size(3))              # (bs , n_ret , L_gen , num_layers , hidden)

            for b in range(bs):
                sample_row = []
                for r in range(n_ret):
                    v = valid_len[b, r].item()
                    sample_row.append(state_cpu[b, r, :v].clone())       # trim to real length
                state.append(sample_row)

        output_line = [dict(zip(output_line, t)) for t in zip(*output_line.values())]
        generation = generation + output_line

    
    save_path = f"{args.save_dir}/{args.dataset}/{args.model}/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path+"generation.pkl", "wb") as f:
        pickle.dump(generation, f)

    if args.logit:
        with open(save_path+"logit.pkl", "wb") as f:
            pickle.dump(logit, f)

    if args.log_prob:
        with open(save_path+"log_prob.pkl", "wb") as f:
            pickle.dump(log_prob, f)
        
    if args.hidden_state:
        with open(save_path+"state.pkl", "wb") as f:
            pickle.dump(state, f)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--random_seed", type=int, default=42
        )

    parser.add_argument(
        "--batch_size", type=int, default=16
        )

    parser.add_argument(
        "--max_length", type=int, default=256,
        )

    parser.add_argument(
        "--temperature", type=float, default=0.0
        )
    
    parser.add_argument(
        "--top_k", type=int, default=0
        )

    parser.add_argument(
        "--top_p", type=float, default=1.0
        )
    
    parser.add_argument(
        "--num_return_sequences", type=int, default=1
        )
    
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b-hf"
        )
    
    parser.add_argument(
        "--hidden_state", dest="hidden_state", default=False, action="store_true"
        )
    
    parser.add_argument(
        "--logit", dest="logit", default=False, action="store_true"
        )
    
    parser.add_argument(
        "--log_prob", dest="log_prob", default=False, action="store_true"
        )
    
    parser.add_argument(
        "--dataset", type=str, default="trivia_qa"
        )

    parser.add_argument(
        "--save_dir", type=str, default="./output"
        )

    parser.add_argument(
        "--huggingface_token", type=str, default=None
        )

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    main()
