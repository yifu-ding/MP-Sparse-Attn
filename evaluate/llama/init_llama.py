import os
# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch

import glob

def resolve_snapshot(model_id):
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    safe_model_id = model_id.replace("/", "--")
    pattern = os.path.join(hf_home, "hub", f"models--{safe_model_id}", "snapshots", "*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No snapshot found for {model_id}")
    return matches[0]  # return first snapshot (usually only one)

def init_llama_model(model_path="", device="cuda"):
    """
    Initialize the Llama model
    Args:
        model_path: local model path
    Returns:
        model: loaded model   
        tokenizer: corresponding tokenizer
    """
    
    # model_path = resolve_snapshot(model_path)

    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=False)
    # model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=False)
    
    # model = model.to(device)
    # model = model.bfloat16()
    # model.eval()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=False)

    # initialize an empty model structure (no memory allocation)
    if '70b' in model_path:
        with init_empty_weights():  # use empty init for 70B; 3B may fail
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=False)


    # automatically shard across multiple GPUs (or set device_map manually)
    model = load_checkpoint_and_dispatch(
        model, model_path,
        device_map="auto",  # or custom dict: { "transformer.h.0": 0, ..., "lm_head": 1 }
        no_split_module_classes=["LlamaDecoderLayer", "Qwen2DecoderLayer"]
    )

    model = model.bfloat16()
    model.eval()
    return model, tokenizer

