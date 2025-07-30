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
    初始化Llama模型
    Args:
        model_path: 本地模型路径
    Returns:
        model: 加载的模型   
        tokenizer: 对应的分词器
    """
    
    # model_path = resolve_snapshot(model_path)

    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=False)
    # model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=False)
    
    # model = model.to(device)
    # model = model.bfloat16()
    # model.eval()

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=False)

    # 初始化空模型结构（不会占用内存）
    if '70b' in model_path:
        with init_empty_weights():  # 70b 用 empty，3b 会报错
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=False)


    # 自动分配到多个 GPU（也可以手动设置 device_map）
    model = load_checkpoint_and_dispatch(
        model, model_path,
        device_map="auto",  # 或自定义 dict: { "transformer.h.0": 0, ..., "lm_head": 1 }
        no_split_module_classes=["LlamaDecoderLayer", "Qwen2DecoderLayer"]
    )

    model = model.bfloat16()
    model.eval()
    return model, tokenizer

