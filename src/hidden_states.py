import argparse
import logging
import torch
import json
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import OfflineEmissionsTracker
from pathlib import Path
from typing import List
from utils import find_subsequence, load_hidden_states, append_to_hdf5_language_group
from constants import (DATASETS, MODELS, LANGUAGES, LLAMA_MODEL_PATH, QWEN_MODEL_PATH, 
                       SYS_PROMPT_FILE, DATA_DIR, LOCAL_RESULTS_DIR, BATCH_SIZE, MAX_NEW_TOKENS, EMISSIONS_DIR,
                       DATASPLITS_GMMLU, DATASPLITS_MKQA, LOCAL_RESULTS_DIR)
import sys
import time
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

def extract_hidden_states_answers_qwen(
    local_model_path: str,
    sys_prompt_file: str,
    input_file: str,
    results_dir: str,
    dataset: str,
    datasplit: str,
    lang: str,
    batch_size: int,
    max_new_tokens: int
):

    logging.info("Extracting hidden states and query answers from Qwen3 8B")

    hidden_states_dir = f"{results_dir}/{dataset}/qwen3_8B/{datasplit}/{lang}_all_tokens_q_and_output_hidden_layers"
    os.makedirs(hidden_states_dir, exist_ok=True)

    answers_file = f"{results_dir}/{dataset}/qwen3_8B/{datasplit}/{lang}_answer.jsonl"
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # ── TOKENIZER & MODEL SETUP ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        logging.warning("No padding token identified. Exiting..")
        sys.exit()

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    terminators = [eos_id]

    # ── LOAD SYSTEM PROMPT ─────────────────────────────────────────────────────
    with open(sys_prompt_file, "r", encoding="utf-8") as pf:
        prompts_cfg = json.load(pf)

    if dataset == "mkqa":
        system_instruction = prompts_cfg[dataset][datasplit][lang]
    else:
        system_instruction = prompts_cfg[dataset][lang]

    # ── LOAD INPUT EXAMPLES ────────────────────────────────────────────────────
    examples = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            if dataset == "mkqa":
                examples.append((item["example_id"], item["queries"][lang]))
            else:
                examples.append((item["sample_id"], item["question"]))

    total_batches = (len(examples) + batch_size - 1) // batch_size
    logging.info(f"\n\nTotal examples: {len(examples)} | Batch size: {batch_size} | Total batches: {total_batches}")
    start_total = time.time()
    
    # ── EXTRACT HIDDEN STATES ──────────────────────────────────────────────────
    with torch.no_grad():
        for batch_num, i in enumerate(
            tqdm(
                range(0, len(examples), batch_size), total=total_batches, desc=f"[{dataset}/{datasplit}/{lang}]", unit="batch"
            ), start=1
        ):
            t_batch_start = time.time()
            logging.info(f"\n\nNew batch for examples: {i} to {i + batch_size-1}")
            batch = examples[i : i + batch_size]
            ids, queries = zip(*batch)

            # (1) Pass queries in a chat template
            input_ids_list = []
            for q in queries:
                msgs = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": q},
                ]
                out = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True)
                input_ids_list.append(out["input_ids"])

            # Pad variable length inputs to the same length
            tok = tokenizer.pad({"input_ids": input_ids_list}, padding=True, return_tensors="pt")
            max_prompt_len = tok["input_ids"].shape[1]
            # Keep the lengths of the (sys+user) prompt
            tok = {k: v.to(model.device) for k, v in tok.items()}
            prompt_lens = tok["attention_mask"].sum(dim=1).tolist()
            logging.info(f"\n(Sys + User) prompt lengths: {prompt_lens}")
            logging.info(f"Max prompt length: {max_prompt_len}")

            # Clear any cuda dead fragments before generation
            torch.cuda.empty_cache()

            # (2) Get prompt + generation hidden states
            # Parameter setting based on best practices: https://huggingface.co/Qwen/Qwen3-8B#best-practices
            gen_out = model.generate(
                **tok,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                pad_token_id=pad_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=False,
            )

            logging.info(f"\ngen_out.hidden_states: {len(gen_out.hidden_states)}") # tuple of length equal to the max number of generated tokens for the batch
            sequences = gen_out.sequences.cpu().tolist() # (batch_size, )
            
            for idx, ex_id in enumerate(ids):

                logging.info(f"\n----- Next example idx: {ex_id}")
                query = queries[idx]  # the asked query
                plen = int(prompt_lens[idx])  # the length of the (sys+user) prompt
                seq = sequences[idx]  # the generated sequence
                logging.info(f"Num sequence tokens: {len(seq)}")
                logging.info(f"seq: {seq}")
                
                # (3) Count query and answer tokens and decode answers
                # a. Find the indices of the query tokens
                logging.info("\nFind query indices:")
                pad_count = max_prompt_len - plen
                prompt_ids = seq[pad_count : max_prompt_len] # the prompt tokens, excluding left padding
                logging.info(f"num prompt_ids: {len(prompt_ids)}")
                # Tokenise the query
                # add_special_tokens=False bc inside the prompt, the query appears without its own special tokens
                q_tokens = tokenizer.encode(query, add_special_tokens=False)
                q_start, q_end = find_subsequence(q_tokens, prompt_ids)
                if q_start == -1:
                    logging.info(f"prompt text: {tokenizer.decode(prompt_ids, skip_special_tokens=False)}")
                    logging.info(f"actual q: {tokenizer.decode(q_tokens, skip_special_tokens=False)}")
                    logging.warning(f"!!!!!!!! Could not locate query tokens for example {ex_id}. Skipping.")
                    continue

                # Make q_start, q_end absolute
                q_start += pad_count
                q_end += pad_count

                # Check that the actual and the found query are the same to confirm q_start + q_end positions
                # skip_special_tokens=False for readability
                logging.info(f"prompt text: {tokenizer.decode(prompt_ids, skip_special_tokens=False)}")
                logging.info(f"actual q: {tokenizer.decode(q_tokens, skip_special_tokens=False)}")
                logging.info(f"found q: {tokenizer.decode(seq[q_start : q_end], skip_special_tokens=False)}")
                logging.info(f"q_tokens: {q_tokens}")
                logging.info(f"seq[{q_start} : {q_end}]: {seq[q_start : q_end]}")  

                # b. Find the indices of the answer tokens & decode the answer
                logging.info("\nFind answer indices + decode answer:")
                gen_ids = seq[max_prompt_len:] # all the generated tokens
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True) # skip_special_tokens=True to get the clean answer
                a_tokens = tokenizer.encode(gen_text, add_special_tokens=False) # tokenise the generated text
                a_start, a_end = find_subsequence(a_tokens, gen_ids)
                if a_start == -1:
                    logging.info(f"generated text: {gen_text}")
                    logging.info(f"actual a: {tokenizer.decode(a_tokens, skip_special_tokens=True)}")
                    logging.warning(f"!!!!!!!! Could not locate answer tokens for example {ex_id}. Skipping.")
                    continue

                # Make a_start, a_end absolute
                a_start += max_prompt_len
                a_end += max_prompt_len

                # Check that the actual and the found generated answer are the same to confirm a_start + a_end positions
                # skip_special_tokens=True for readability
                logging.info(f"generated text: {gen_text}")
                logging.info(f"actual a: {tokenizer.decode(a_tokens, skip_special_tokens=True)}")
                logging.info(f"found a: {tokenizer.decode(seq[a_start : a_end], skip_special_tokens=True)}")
                logging.info(f"a_tokens: {a_tokens}")
                logging.info(f"seq[{a_start} : {a_end}]: {seq[a_start : a_end]}") 

                # (4) Extract the per-layer query & answer hidden states
                # gen_out.hidden_states is indexed by generation step
                # gen_hs_all = gen_out.hidden_states[num_gen_steps-1][1:]  # tuple: (num_layers, ), gen_hs_all[layer_i]: (batch_size, max_prompt_len, num_hidden_dims)
                # For both query + answer: Keep until the last context token of the query, irrespective of what it is e.g., word, fullstop, question mark
                logging.info("\nSlice out per-layer query hidden states:")
    
                # a. Get the example's query hidden states
                # Drop system prompt tokens + special tokens added after the user query
                query_hidden_states = []
                # gen_out.hidden_states[0]: the hidden states after the entire prompt processing
                for layer_hs in gen_out.hidden_states[0][1:]: 
                    q_hs = layer_hs[idx, q_end - 1, :].cpu()  # (hidden_dim,)
                    query_hidden_states.append(q_hs)
                    
                # b. Get answer hidden states
                # Drop special tokens added after the generated answer
                answer_hidden_states = []
                computed_step = a_end - max_prompt_len
                last_answer_step = min(computed_step, len(gen_out.hidden_states) - 1)
                if computed_step > last_answer_step:
                    logging.warning(f"Example {ex_id}: answer hit max_new_tokens, last token hidden state unavailable, truncating.")
                # Clamp the token index to the last available position at last_answer_step
                effective_last_token = min(a_end - 1, max_prompt_len + last_answer_step - 1)
                for layer_hs in gen_out.hidden_states[last_answer_step][1:]:
                    a_hs = layer_hs[idx, effective_last_token, :].cpu()  # (hidden_dim,)
                    answer_hidden_states.append(a_hs)
                
                logging.info("Example summary:")
                logging.info(f"\tTotal tokens: {layer_hs[idx, :, :].shape}")
                # query 
                logging.info(f"\t(Sys+User) prompt tokens: {plen} out of which:") 
                logging.info(f"\t\tPad tokens: {pad_count}") 
                logging.info(f"\t\tQuery tokens: {len(seq[q_start : q_end])}")
                logging.info(f"\t\tKept query tokens: {q_hs.shape}") 
                # answer
                logging.info(f"\tTotal generation tokens: {len(gen_ids)} out of which:") 
                logging.info(f"\t\tAnswer tokens: {len(seq[a_start : a_end])}")
                logging.info(f"\t\tKept answer tokens: {a_hs.shape}") 

                # (5) Save results
                # Write hidden states
                if dataset == "global_mmlu":
                    safe_id = ex_id.replace("/", "_")
                    hs_path = os.path.join(hidden_states_dir, f"{safe_id}.pt")
                else:
                    hs_path = os.path.join(hidden_states_dir, f"{ex_id}.pt")
                torch.save({"query_hidden_states": torch.stack(query_hidden_states, dim=0), 
                            "answer_hidden_states": torch.stack(answer_hidden_states, dim=0)}, 
                            hs_path)

                # Write the answer record
                rec = {
                    "example_id": ex_id,
                    "query": query,
                    "q_start": q_start,
                    "q_end": q_end,
                    "a_start": a_start,
                    "a_end": a_end, 
                    "answer": gen_text,
                }
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                with open(answers_file, "a", encoding="utf-8") as ans_f:
                    ans_f.write(line)

            # ── PER-BATCH TIMING ──────────────────────────────────────────────
            t_batch_elapsed = time.time() - t_batch_start
            t_total_elapsed = time.time() - start_total
            avg_per_batch = t_total_elapsed / batch_num
            remaining_batches = total_batches - batch_num
            eta_seconds = avg_per_batch * remaining_batches

            logging.info(
                f"\nBatch {batch_num}/{total_batches} | "
                f"This batch: {t_batch_elapsed:.1f}s | "
                f"Avg/batch: {avg_per_batch:.1f}s | "
                f"ETA: {eta_seconds / 60:.1f} min"
            )
            
            # ── MEMORY CLEANUP ──────────────────────────────────────────────
            try:
                del gen_out, tok, sequences
                if 'query_hidden_states' in locals():
                    del query_hidden_states
                if 'answer_hidden_states' in locals():
                    del answer_hidden_states
            except Exception:
                pass  # best-effort cleanup; don't skip cuda release on error
            torch.cuda.empty_cache()
            gc.collect()

            logging.info(f"Processed batch {i//batch_size + 1}")

    # ── MERGE PER-EXAMPLE FILES INTO ONE ──────────────────────────────────────
    logging.info("Merging per-example hidden state files into one...")
    dir_path = Path(hidden_states_dir)
    out_file = Path(f"{results_dir}/{dataset}/qwen3_8B/{datasplit}/{lang}_all_tokens_q_and_output_hidden_layers.pt")
    all_hs = {}
    for path in sorted(dir_path.glob("*.pt")):
        ex_id = path.stem
        hs = torch.load(path, map_location="cpu")
        try:
            key = int(ex_id)
        except ValueError:
            key = ex_id
        all_hs[key] = {k: v.cpu() for k, v in hs.items()}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_hs, out_file)

    # ── CLEAN UP PER-EXAMPLE FILES ────────────────────────────────────────────
    logging.info("Cleaning up per-example hidden state files...")
    n_deleted = 0
    for path in dir_path.glob("*.pt"):
        try:
            path.unlink()
            n_deleted += 1
        except Exception as e:
            logging.warning(f"Could not delete {path}: {e}")
    try:
        dir_path.rmdir()  # only succeeds if the directory is now empty
        logging.info(f"Removed directory {dir_path}")
    except Exception as e:
        logging.warning(f"Could not remove directory {dir_path}: {e}")
    logging.info(f"Deleted {n_deleted} per-example files")
    
    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    total_time = time.time() - start_total
    logging.info(
        f"\nDone. {len(examples)} examples in {total_batches} batches | "
        f"Total time: {total_time / 60:.1f} min ({total_time / 3600:.2f} h)"
    )
    logging.info(f"✅ Merged hidden states saved to {out_file}")
    logging.info(f"✅ Saved model answers to {answers_file}")


def extract_hidden_states_answers_llama(
    local_model_path: str,
    sys_prompt_file: str,
    input_file: str,
    results_dir: str,
    dataset: str,
    datasplit: str,
    lang: str,
    batch_size: int,
    max_new_tokens: int
):
    
    logging.info("Extracting hidden states and query answers from Llama 3.1 8B Instruct")

    hidden_states_dir = f"{results_dir}/{dataset}/llama_3.1_8B/{datasplit}/{lang}_all_tokens_q_and_output_hidden_layers"
    os.makedirs(hidden_states_dir, exist_ok=True)

    answers_file = f"{results_dir}/{dataset}/llama_3.1_8B/{datasplit}/{lang}_answer.jsonl"

    # ── TOKENIZER & MODEL SETUP ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        use_fast=True,
        padding_side="left",
    )
    # logging.info("Tokenizer class:", tokenizer.__class__.__name__)

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    # logging.info("Model class:", model.__class__.__name__)

    # Set EOS as pad
    tokenizer.pad_token_id = tokenizer.eos_token_id
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    # ── LOAD SYSTEM PROMPT ─────────────────────────
    with open(sys_prompt_file, "r", encoding="utf-8") as pf:
        prompts_cfg = json.load(pf)
    
    if dataset == "mkqa":
        system_instruction = prompts_cfg[dataset][datasplit][lang]
    else:
        system_instruction = prompts_cfg[dataset][lang]

    # ── LOAD INPUT EXAMPLES ───────────────────────────────────────────────────
    examples = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            if dataset == "mkqa":
                examples.append((item["example_id"], item["queries"][lang]))
            else:
                examples.append((item["sample_id"], item["question"]))
    
    total_batches = (len(examples) + batch_size - 1) // batch_size
    logging.info(f"\n\nTotal examples: {len(examples)} | Batch size: {batch_size} | Total batches: {total_batches}")
    start_total = time.time()
    
    # ── EXTRACT HIDDEN STATES ──────────────────────────────────────────────────
    with torch.no_grad():
        for batch_num, i in enumerate(
            tqdm(
                range(0, len(examples), batch_size), total=total_batches, desc=f"[{dataset}/{datasplit}/{lang}]", unit="batch"
            ), start=1
        ):
            t_batch_start = time.time()
            logging.info(f"\n\nNew batch for examples: {i} to {i + batch_size-1}")
            batch = examples[i : i + batch_size]
            ids, queries = zip(*batch)

            # (1) Pass queries in a chat template
            input_ids = []
            for q in queries:
                msgs = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": q},
                ]
                full_input_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
                input_ids.append(full_input_ids)

            # Pad variable length inputs to the same length
            # tok: dict with:
            # "input_ids": a 2D PyTorch Tensor of shape (batch_size, max_seq_len) with your token IDs, padded to the same length
            # "attention_mask": a 2D PyTorch Tensor of shape (batch_size, max_seq_len) of 1s (real tokens) and 0s (padding)
            tok = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
            max_prompt_len = tok["input_ids"].shape[1]
            # Move tokenised input to device
            tok = {k: v.to(model.device) for k, v in tok.items()}
            # Keep the lengths of the (sys+user) prompt
            prompt_lens = tok["attention_mask"].sum(dim=1).tolist() # (batch_size, )
            logging.info(f"\n(Sys + User) prompt lengths: {prompt_lens}")
            logging.info(f"Max prompt length: {max_prompt_len}")

            # Clear any cuda dead fragments before generation
            torch.cuda.empty_cache()

            # (2) Get prompt + generation hidden states with dflt generation settings, excluding the embedding layer
            # https://huggingface.co/docs/transformers/v5.7.0/en/main_classes/text_generation#transformers.GenerationMixin.generate
            gen_out = model.generate(
                **tok,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=False,
            )            
            
            logging.info(f"\ngen_out.hidden_states: {len(gen_out.hidden_states)}") # tuple of length equal to the max number of generated tokens for the batch
            sequences = gen_out.sequences.cpu().tolist() # (batch_size, ) 
            
            for idx, ex_id in enumerate(ids):

                logging.info(f"\n----- Next example idx: {ex_id}")
                query = queries[idx] # the asked query
                plen = prompt_lens[idx] # the length of the (sys+user) prompt
                seq = sequences[idx] # the generated sequence
                logging.info(f"Num sequence tokens: {len(seq)}")
                logging.info(f"seq: {seq}")
                # logging.info(f"Full sequence: {tokenizer.decode(seq, skip_special_tokens=True)}") # include both the prompt + the answer (n-1) hidden states

                # (3) Count query and answer tokens and decode answers
                # a. Find the indices of the query tokens
                logging.info("\nFind query indices:")
                pad_count = max_prompt_len - plen
                prompt_ids = seq[pad_count : max_prompt_len] # the prompt tokens, excluding left padding
                logging.info(f"num prompt_ids: {len(prompt_ids)}")
                # Tokenise the query
                # add_special_tokens=False bc inside the prompt, the query appears without its own special tokens
                q_tokens = tokenizer.encode(query, add_special_tokens=False)
                q_start, q_end = find_subsequence(q_tokens, prompt_ids)
                if q_start == -1:
                    logging.info(f"prompt text: {tokenizer.decode(prompt_ids, skip_special_tokens=False)}")
                    logging.info(f"actual q: {tokenizer.decode(q_tokens, skip_special_tokens=False)}")
                    logging.warning(f"!!!!!!!! Could not locate query tokens for example {ex_id}. Skipping.")
                    continue

                # Make q_start, q_end absolute
                q_start += pad_count
                q_end += pad_count
                
                # Check that the actual and the found query are the same to confirm q_start + q_end positions
                # skip_special_tokens=False for readability
                logging.info(f"prompt text: {tokenizer.decode(prompt_ids, skip_special_tokens=False)}")
                logging.info(f"actual q: {tokenizer.decode(q_tokens, skip_special_tokens=False)}")
                logging.info(f"found q: {tokenizer.decode(seq[q_start : q_end], skip_special_tokens=False)}")
                logging.info(f"q_tokens: {q_tokens}")
                logging.info(f"seq[{q_start} : {q_end}]: {seq[q_start : q_end]}") 

                # b. Find the indices of the answer tokens & decode the answer
                logging.info("\nFind answer indices + decode answer:")
                gen_ids = seq[max_prompt_len:] # all the generated tokens
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True) # skip_special_tokens=True to get the clean answer
                a_tokens = tokenizer.encode(gen_text, add_special_tokens=False) # tokenise the generated text
                a_start, a_end = find_subsequence(a_tokens, gen_ids)
                if a_start == -1:
                    logging.info(f"generated text: {gen_text}")
                    logging.info(f"actual a: {tokenizer.decode(a_tokens, skip_special_tokens=True)}")
                    logging.warning(f"!!!!!!!! Could not locate answer tokens for example {ex_id}. Skipping.")
                    continue

                # Make a_start, a_end absolute
                a_start += max_prompt_len
                a_end += max_prompt_len
                
                # Check that the actual and the found generated answer are the same to confirm a_start + a_end positions
                # skip_special_tokens=True for readability
                logging.info(f"generated text: {gen_text}")
                logging.info(f"actual a: {tokenizer.decode(a_tokens, skip_special_tokens=True)}")
                logging.info(f"found a: {tokenizer.decode(seq[a_start : a_end], skip_special_tokens=True)}")
                logging.info(f"a_tokens: {a_tokens}")
                logging.info(f"seq[{a_start} : {a_end}]: {seq[a_start : a_end]}") 

                # (4) Extract the per-layer query & answer hidden states
                # gen_out.hidden_states is indexed by generation step
                # gen_hs_all = gen_out.hidden_states[num_gen_steps-1][1:]  # tuple: (num_layers, ), gen_hs_all[layer_i]: (batch_size, max_prompt_len, num_hidden_dims)
                # For both query + answer: Keep until the last context token of the query, irrespective of what it is e.g., word, fullstop, question mark
                logging.info("\nSlice out per-layer query hidden states:")
    
                # a. Get the example's query hidden states
                # Drop system prompt tokens + special tokens added after the user query
                query_hidden_states = []
                # gen_out.hidden_states[0]: the hidden states after the entire prompt processing
                for layer_hs in gen_out.hidden_states[0][1:]: 
                    q_hs = layer_hs[idx, q_end - 1, :].cpu()  # (hidden_dim,)
                    query_hidden_states.append(q_hs)
                    
                # b. Get answer hidden states
                # Drop special tokens added after the generated answer
                answer_hidden_states = []
                computed_step = a_end - max_prompt_len
                last_answer_step = min(computed_step, len(gen_out.hidden_states) - 1)
                if computed_step > last_answer_step:
                    logging.warning(f"Example {ex_id}: answer hit max_new_tokens, last token hidden state unavailable, truncating.")
                # Clamp the token index to the last available position at last_answer_step
                effective_last_token = min(a_end - 1, max_prompt_len + last_answer_step - 1)
                for layer_hs in gen_out.hidden_states[last_answer_step][1:]:
                    a_hs = layer_hs[idx, effective_last_token, :].cpu()  # (hidden_dim,)
                    answer_hidden_states.append(a_hs)
                
                logging.info("Example summary:")
                logging.info(f"\tTotal tokens: {layer_hs[idx, :, :].shape}")
                # query 
                logging.info(f"\t(Sys+User) prompt tokens: {plen} out of which:") 
                logging.info(f"\t\tPad tokens: {pad_count}") 
                logging.info(f"\t\tQuery tokens: {len(seq[q_start : q_end])}")
                logging.info(f"\t\tKept query tokens: {q_hs.shape}") 
                # answer
                logging.info(f"\tTotal generation tokens: {len(gen_ids)} out of which:") 
                logging.info(f"\t\tAnswer tokens: {len(seq[a_start : a_end])}")
                logging.info(f"\t\tKept answer tokens: {a_hs.shape}") 

                # (5) Save results
                # Write hidden states
                if dataset == "global_mmlu":
                    safe_id = ex_id.replace("/", "_")
                    hs_path = os.path.join(hidden_states_dir, f"{safe_id}.pt")
                else:
                    hs_path = os.path.join(hidden_states_dir, f"{ex_id}.pt")
                torch.save({"query_hidden_states": torch.stack(query_hidden_states, dim=0), 
                            "answer_hidden_states": torch.stack(answer_hidden_states, dim=0)}, 
                            hs_path)

                # Write the answer record
                rec = {
                    "example_id": ex_id,
                    "query": query,
                    "q_start": q_start,
                    "q_end": q_end,
                    "a_start": a_start,
                    "a_end": a_end, 
                    "answer": gen_text,
                }
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                with open(answers_file, "a", encoding="utf-8") as ans_f:
                    ans_f.write(line)

            # ── PER-BATCH TIMING ──────────────────────────────────────────────
            t_batch_elapsed = time.time() - t_batch_start
            t_total_elapsed = time.time() - start_total
            avg_per_batch = t_total_elapsed / batch_num
            remaining_batches = total_batches - batch_num
            eta_seconds = avg_per_batch * remaining_batches

            logging.info(
                f"\nBatch {batch_num}/{total_batches} | "
                f"This batch: {t_batch_elapsed:.1f}s | "
                f"Avg/batch: {avg_per_batch:.1f}s | "
                f"ETA: {eta_seconds / 60:.1f} min"
            )
            
            # ── MEMORY CLEANUP ──────────────────────────────────────────────
            try:
                del gen_out, tok, sequences
                if 'query_hidden_states' in locals():
                    del query_hidden_states
                if 'answer_hidden_states' in locals():
                    del answer_hidden_states
            except Exception:
                pass  # best-effort cleanup; don't skip cuda release on error
            torch.cuda.empty_cache()
            gc.collect()

            logging.info(f"Processed batch {i//batch_size + 1}")

    # ── MERGE PER-EXAMPLE FILES INTO ONE ──────────────────────────────────────
    logging.info("Merging per-example hidden state files into one...")
    dir_path = Path(hidden_states_dir)
    out_file = Path(f"{results_dir}/{dataset}/llama_3.1_8B/{datasplit}/{lang}_all_tokens_q_and_output_hidden_layers.pt")
    all_hs = {}
    for path in sorted(dir_path.glob("*.pt")):
        ex_id = path.stem
        hs = torch.load(path, map_location="cpu")
        try:
            key = int(ex_id)
        except ValueError:
            key = ex_id
        all_hs[key] = {k: v.cpu() for k, v in hs.items()}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_hs, out_file)

    # ── CLEAN UP PER-EXAMPLE FILES ────────────────────────────────────────────
    logging.info("Cleaning up per-example hidden state files...")
    n_deleted = 0
    for path in dir_path.glob("*.pt"):
        try:
            path.unlink()
            n_deleted += 1
        except Exception as e:
            logging.warning(f"Could not delete {path}: {e}")
    try:
        dir_path.rmdir()  # only succeeds if the directory is now empty
        logging.info(f"Removed directory {dir_path}")
    except Exception as e:
        logging.warning(f"Could not remove directory {dir_path}: {e}")
    logging.info(f"Deleted {n_deleted} per-example files")
    
    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    total_time = time.time() - start_total
    logging.info(
        f"\nDone. {len(examples)} examples in {total_batches} batches | "
        f"Total time: {total_time / 60:.1f} min ({total_time / 3600:.2f} h)"
    )
    logging.info(f"✅ Merged hidden states saved to {out_file}")
    logging.info(f"✅ Saved model answers to {answers_file}")


def save_hidden_states_from_dir_to_file(lang, results_dir, device="cpu"):
    """
    Collect all per-example hidden-state files into one dict and save it.

    Args:
        lang (str): e.g. "en"
        results_dir (str | Path): parent results directory that contains
            `<lang>_all_tokens_q_and_output_hidden_layers/`
        out_file (str | Path | None): where to save the merged dict.
            If None, defaults to `<results_dir>/<lang>_all_hidden_states.pt`
        device (str): map_location for torch.load

    Returns:
        Path: path to the written .pt file
    """
    dir_path = Path(results_dir) / f"{lang}_all_tokens_q_and_output_hidden_layers"
    out_file = Path(results_dir) / f"{lang}_all_tokens_q_and_output_hidden_layers.pt"

    all_hs = {}
    for path in sorted(dir_path.glob("*.pt")):
        ex_id = path.stem
        hs = torch.load(path, map_location=device)   # {"query_hidden_states": ..., "gen_hidden_states": ...}
        # make sure tensors are on CPU so the file is portable/smaller
        all_hs[int(ex_id)] = {k: v.cpu() for k, v in hs.items()}

    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_hs, out_file)


def merge_hidden_states(datasplits: List[str], languages: List[str], results_dir: str):

    # Set the output paths
    hdf5_query_path = Path(results_dir) / f"query_last_token_hidden_states.h5"
    hdf5_answer_path = Path(results_dir) / f"answer_last_token_hidden_states.h5"

    for ds in datasplits:
        results_dir_ds = Path(results_dir) / ds
        logging.info(f"results_dir_ds: {results_dir_ds}")
        for lang in languages:
            hidden_states_lang_ds = load_hidden_states(lang, results_dir_ds)
            q_last_token_lang_ds = {}
            a_last_token_lang_ds = {}

            # Extract the hidden states
            for eid, hs_dict in hidden_states_lang_ds.items():
                q_last_token_lang_ds[eid] = hs_dict['query_hidden_states']
                a_last_token_lang_ds[eid] = hs_dict['answer_hidden_states']

            # Append to separate persistent HDF5 files under language groups (dataset info dropped)
            try:
                append_to_hdf5_language_group(hdf5_query_path, lang, q_last_token_lang_ds)
            except Exception as e:
                logging.warning(f"Failed to persist query states for lang={lang}, dataset={ds}: {e}")

            try:
                append_to_hdf5_language_group(hdf5_answer_path, lang, a_last_token_lang_ds)
            except Exception as e:
                logging.warning(f"Failed to persist answer states for lang={lang}, dataset={ds}: {e}")

            # Cleanup
            del hidden_states_lang_ds, q_last_token_lang_ds, a_last_token_lang_ds
            gc.collect()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-emissions", action="store_true", default=False)

    sub = parser.add_subparsers(dest="operation", required=True)

    p_extract = sub.add_parser("extract_hidden_states")
    p_extract.add_argument("--model-name", type=str, choices=MODELS, required=True)
    p_extract.add_argument("--dataset-name", type=str, choices=DATASETS, required=True)
    p_extract.add_argument("--datasplit-name", type=str, choices=DATASPLITS_GMMLU + DATASPLITS_MKQA, required=True)
    p_extract.add_argument("--lang", type=str, choices=LANGUAGES, required=True)

    p_merge = sub.add_parser("merge_hidden_states")
    p_merge.add_argument("--model-name", type=str, choices=MODELS, required=True)
    p_merge.add_argument("--dataset-name", type=str, choices=DATASETS, required=True)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    tracker = None
    if args.track_emissions:
        out_file = (
            "03_extract_hidden_states_es.csv"
            if args.operation == "extract_hidden_states"
            else "merge_hidden_states.csv"
        )
        tracker = OfflineEmissionsTracker(output_dir=EMISSIONS_DIR, output_file=out_file)
        tracker.start()

    try:
        if args.operation == "extract_hidden_states":
            dataset = args.dataset_name
            datasplit = args.datasplit_name
            lang = args.lang
            model = args.model_name

            if dataset == "mkqa":
                input_file = f"{DATA_DIR}/{dataset}/{datasplit}.jsonl"
                max_new_tokens = 2 if datasplit == "mkqa_answerable_binary" and lang == "de" else (
                    1 if datasplit == "mkqa_answerable_binary" else MAX_NEW_TOKENS
                )
            elif dataset == "global_mmlu":
                input_file = f"{DATA_DIR}/{dataset}/{lang}_final/{datasplit}.jsonl"
                max_new_tokens = MAX_NEW_TOKENS
            else:
                logging.error("Dataset not integrated")
                sys.exit()

            if model == "llama_3.1_8B":
                extract_hidden_states_answers_llama(
                    LLAMA_MODEL_PATH, SYS_PROMPT_FILE, input_file,
                    LOCAL_RESULTS_DIR, dataset, datasplit, lang, BATCH_SIZE, max_new_tokens
                )
            
            elif model == "qwen3_8B":
                extract_hidden_states_answers_qwen(
                    QWEN_MODEL_PATH, SYS_PROMPT_FILE, input_file,
                    LOCAL_RESULTS_DIR, dataset, datasplit, lang, BATCH_SIZE, max_new_tokens
                )

        elif args.operation == "merge_hidden_states":
            dataset = args.dataset_name
            datasplits = DATASPLITS_MKQA if dataset == 'mkqa' else DATASPLITS_GMMLU
            model = args.model_name
            res_dir = Path(LOCAL_RESULTS_DIR) / dataset / model
            merge_hidden_states(datasplits, LANGUAGES, res_dir)

    finally:
        if tracker is not None:
            tracker.stop()