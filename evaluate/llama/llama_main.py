import sys
import os

# 计算项目根路径（假设项目结构：MP-Sparse-Attn/evaluate/llama/llama_main.py）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 确保项目根目录在 sys.path 最前面，优先导入
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from init_llama import init_llama_model
import argparse
# from modify_sparse_attn import set_spas_sage_attn_llama
# from spas_sage_attn.autotune import (
#     extract_sparse_attention_state_dict,
#     load_sparse_attention_state_dict,
# )
import torch
import numpy as np
import random
import json
import gc
from tqdm import tqdm
import wandb

from ours.modify_mxfp_attn import set_mxfp_attn_llama
from evaluate.datasets.text.longbench.pred import build_chat, get_pred, get_pred_speedup
from datasets import load_dataset
from evaluate.datasets.text.longbench.eval import scorer_e, scorer
# import torch.multiprocessing as mp
from tests.modify_flash_attn_triton import set_flash_attn_triton_llama

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description='Llama model parameters')
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda',
                      help='running device (cuda/cpu)')
    
    # 稀疏注意力相关参数  
    parser.add_argument('--tune', action='store_true', help="whether to tune")
    parser.add_argument('--verbose', action='store_true', help="whether to print verbose")
    parser.add_argument('--l1', type=float, default=0.06, help='l1 bound for qk sparse')
    parser.add_argument('--pv_l1', type=float, default=0.065, help='l1 bound for pv sparse')

    # dataset 
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--test_dataset_name', type=str, default='all', help="Evaluate on single dataset")

    # output path
    parser.add_argument('--output_path', type=str, default="results",
                      help='output path')
    parser.add_argument('--model_out_path', type=str, default="", help="Model out path") # save tuned model state_dict
    
    parser.add_argument('--use_wandb', action='store_true', help="use wandb")
    parser.add_argument('--num_fewshots', type=int, default=None, help="number of fewshots")
    
    # test config
    parser.add_argument('--compute_accuracy', action='store_true', help="compute accuracy")
    parser.add_argument('--get_pred', action='store_true', help="get pred")
    parser.add_argument('--test_speedup', action='store_true', help="test speedup")
    
    # our method
    parser.add_argument('--skip_thresh', type=float, default=None, help="skip threshold")
    parser.add_argument('--kernel_name', type=str, default=None, help="kernel name", choices=["online_routing", "mxfp_attn", "mxfp_attn_debug", "native", 'spargeattn'])
    parser.add_argument('--mxfp_bw', type=str, default='mxfp8', help="mxfp bw")
    parser.add_argument('--smooth_k', action='store_true', help="smooth k")
    parser.add_argument('--dual_scale', action='store_true', help="dual scale")

    parser.add_argument('--pre_quant', type=bool, default=False, help="pre quant")
    parser.add_argument('--fuse_mp_quant', type=bool, default=False, help="fuse mp quant")
    parser.add_argument('--fp8_tile_num', type=int, default=1, help="fp8 tile num")

    # 解析参数
    args = parser.parse_args()
    assert args.compute_accuracy or args.test_speedup or args.get_pred, "must choose one of compute_accuracy or test_speedup or get_pred"
    assert args.fp8_tile_num > 0, "fp8_tile_num must be greater than 0"

    print("***** args *****")
    print(f"    - model: {args.model}")
    print(f"    - device: {args.device}")
    print(f"    - kernel_name: {args.kernel_name}")
    print(f"    - verbose: {args.verbose}")
    print(f"    - num_fewshots: {args.num_fewshots}")
    print(f"    - output_path: {args.output_path}")
    print(f"    - model_out_path: {args.model_out_path}")
    print(f"    - compute_accuracy: {args.compute_accuracy}")
    print(f"    - get_pred: {args.get_pred}")
    print(f"    - test_speedup: {args.test_speedup}")
    print(f"    - skip_thresh: {args.skip_thresh}")
    print(f"    - test_dataset_name: {args.test_dataset_name}")
    if "mxfp_attn" in args.kernel_name:
        print(f"    - mxfp_bw: {args.mxfp_bw}")
        print(f"    - smooth_k: {args.smooth_k}")
        print(f"    - dual_scale: {args.dual_scale}")
    elif "spargeattn" in args.kernel_name:
        print(f"    - l1: {args.l1}")
        print(f"    - pv_l1: {args.pv_l1}")
        print(f"    - tune: {args.tune}")
    print("*" * 20)
    
    device = torch.device(args.device)
    args.model = args.model.replace("/", "--").lower()
    
    # initialize wandb
    if args.use_wandb:
        if args.kernel_name == "online_routing":
            wandb_name = f"{args.model}-{args.kernel_name}-{args.skip_thresh}-{args.num_fewshots}shots-{args.test_dataset_name}"
        elif "spargeattn" in args.kernel_name:
            wandb_name = f"{args.model}-{args.kernel_name}-{args.l1}-{args.pv_l1}-{args.num_fewshots}shots-{args.test_dataset_name}"
        elif args.kernel_name == "mxfp_attn":
            wandb_name = f"{args.model}-{args.kernel_name}-{args.mxfp_bw}-{args.num_fewshots}shots-{args.test_dataset_name}"
        else:
            wandb_name = f"{args.model}-naive-{args.num_fewshots}shots-{args.test_dataset_name}"
        run = wandb.init(project='mp-sparse', name=wandb_name, entity='eveedyf-google')
    
    # define model and tokenizer
    model2path = json.load(open("evaluate/datasets/text/longbench/config/model2path.json", "r"))
    model, tokenizer = init_llama_model(model_path=model2path[args.model], device=device)
    
    model2maxlen = json.load(open("evaluate/datasets/text/longbench/config/model2maxlen.json", "r"))
    model_name = args.model
    # define datasets
    max_length = model2maxlen[model_name] # max generation length
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    if args.test_dataset_name != "all":  # only test on one dataset
        datasets = [args.test_dataset_name]
    
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("evaluate/datasets/text/longbench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("evaluate/datasets/text/longbench/config/dataset2maxlen.json", "r"))
    
    output_path = args.output_path
    out_path_pred = os.path.join(output_path, "pred")
    out_path_pred_e = os.path.join(output_path, "pred_e")
    # predict on each dataset
    if not os.path.exists(out_path_pred):
        os.makedirs(out_path_pred)
    if not os.path.exists(out_path_pred_e):
        os.makedirs(out_path_pred_e)
        
    # finetune the model with sparse attention, store the state_dict
    if "spargeattn" in args.kernel_name:
        model_out_path = args.model_out_path    
        
        if args.tune:
            os.environ["TUNE_MODE"] = "1"  # enable tune mode

            # 设置稀疏注意力并进行tune
            set_spas_sage_attn_llama(model, verbose=args.verbose, l1=args.l1, pv_l1=args.pv_l1, kernel_name=args.kernel_name)
            print("replace sparse attention and start tune!")
            
            # tune_dataset = "qasper"
            for tune_dataset in tqdm(datasets, desc="Tuning datasets"):
                # 准备一些示例数据进行tune
                tune_data = load_dataset(
                    "THUDM/LongBench",
                    tune_dataset,
                    split="test",
                    trust_remote_code=True
                )
                
                # 进行tune
                prompt_format = dataset2prompt[tune_dataset]
                max_gen = dataset2maxlen[tune_dataset]
                # print("start to tune...")
                for json_obj in tqdm(list(tune_data)[:5], desc="Samples"):  # 每个数据集 n 个 作为 tune sample
                    prompt = prompt_format.format(**json_obj)
                    
                    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                    if "chatglm3" in model_name:
                        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
                    if len(tokenized_prompt) > max_length:
                        half = int(max_length/2)
                        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                    if tune_dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                        prompt = build_chat(tokenizer, prompt, model_name)
                    if "chatglm3" in model_name:
                        if tune_dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                        else:
                            input = prompt.to(device)
                    else:
                        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                    context_length = input.input_ids.shape[-1]
                    
                    if tune_dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                        output = model.generate(
                            **input,
                            max_new_tokens=1,
                            num_beams=1,
                            do_sample=False,
                            top_p=None,  # 显式设置 top_p 为 None
                            temperature=1.0,
                            min_length=context_length+1,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                        )
                    else:
                        output = model.generate(
                            **input,
                            max_new_tokens=1,  # no need to generate tokens
                            num_beams=1,
                            top_p=None,  # 显式设置 top_p 为 None   
                            do_sample=False,
                            temperature=1.0,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    # 清理内存
                    del output
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # 保存tune后的state_dict
            saved_state_dict = extract_sparse_attention_state_dict(model)
            torch.save(saved_state_dict, model_out_path)
            print(f"Tune completed, model state dict saved to: {model_out_path}")
        else:
            # 加载之前保存的state_dict
            if os.path.exists(model_out_path):
                saved_state_dict = torch.load(model_out_path)
                set_spas_sage_attn_llama(model, verbose=args.verbose, l1=args.l1, pv_l1=args.pv_l1, kernel_name=args.kernel_name)
                load_sparse_attention_state_dict(model, saved_state_dict)
                print("replace sparge attention and load state dict!")
            else:
                raise ValueError(f"not find tuned model state dict: {model_out_path}")
    elif args.kernel_name == "online_routing":
        set_spas_sage_attn_llama(model, verbose=args.verbose, skip_thresh=args.skip_thresh, kernel_name=args.kernel_name)
        print("replace outline_routing!")
    elif "mxfp_attn" in args.kernel_name:
        set_mxfp_attn_llama(model, verbose=args.verbose, kernel_name=args.kernel_name, mxfp_bw=args.mxfp_bw, \
            smooth_k=args.smooth_k, dual_scale=args.dual_scale, pre_quant=args.pre_quant, fuse_mp_quant=args.fuse_mp_quant,
            fp8_tile_num=args.fp8_tile_num)
        print(f"replace mxfp_attn {args.mxfp_bw}!")
    elif "flash_attn" in args.kernel_name:
        set_flash_attn_triton_llama(model, smooth_k=args.smooth_k)
        print("replace flash_attn!")
    elif args.kernel_name == "native":
        print("use the original transformer attention!")
    else:
        raise ValueError(f"Unknown kernel name: {args.kernel_name}")
        
        
    model.eval()  # 设置为评估模式
    os.environ["TUNE_MODE"] = "0"  # disable tune mode
    world_size = torch.cuda.device_count()
    # mp.set_start_method('spawn', force=True)
    
    if args.test_speedup:
        for dataset in datasets:
            if args.e:
                data = load_dataset(
                    "THUDM/LongBench",
                    dataset,
                    split="test",
                    trust_remote_code=True
                )
                if not os.path.exists(os.path.join(out_path_pred_e, model_name)):
                    os.makedirs(os.path.join(out_path_pred_e, model_name))
                out_path = os.path.join(out_path_pred_e, model_name, f"{dataset}.jsonl")
            else:
                data = load_dataset(
                    "THUDM/LongBench",
                    dataset,
                    split="test",
                    trust_remote_code=True
                )
                if not os.path.exists(os.path.join(out_path_pred, model_name)):
                    os.makedirs(os.path.join(out_path_pred, model_name))
                out_path = os.path.join(out_path_pred, model_name, f"{dataset}.jsonl")
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            time_per_sample = get_pred_speedup(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, out_path, num_fewshots=args.num_fewshots)
            if args.use_wandb:
                wandb.log({"time_per_sample": time_per_sample})

    if args.get_pred:
        for dataset in datasets:
            if args.e:
                data = load_dataset(
                    "THUDM/LongBench",
                    dataset,
                    split="test",
                    trust_remote_code=True
                )
                if not os.path.exists(os.path.join(out_path_pred_e, model_name)):
                    os.makedirs(os.path.join(out_path_pred_e, model_name))
                out_path = os.path.join(out_path_pred_e, model_name, f"{dataset}.jsonl")
            else:
                data = load_dataset(
                    "THUDM/LongBench",
                    dataset,
                    split="test",
                    trust_remote_code=True
                )
                if not os.path.exists(os.path.join(out_path_pred, model_name)):
                    os.makedirs(os.path.join(out_path_pred, model_name))
                out_path = os.path.join(out_path_pred, model_name, f"{dataset}.jsonl")
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            # data_all = [data_sample for data_sample in data]
            # data_subsets = [data_all[i::world_size] for i in range(world_size)]
            # processes = []
            # for rank in range(world_size):
            #     p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
            #                 max_gen, prompt_format, dataset, device, model_name, model2path, out_path))
            #     p.start()
            #     processes.append(p)
            # for p in processes:
            #     p.join()
            
            # def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, out_path):
            get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, out_path, num_fewshots=args.num_fewshots)

    if args.compute_accuracy:
        scores = dict()
        if args.e:
            path = os.path.join(out_path_pred_e, model_name)
        else:
            path = os.path.join(out_path_pred, model_name)
        all_files = os.listdir(path)
        print("Evaluating on:", all_files)
        for filename in all_files:
            if not filename.endswith("jsonl"):
                continue
            predictions, answers, lengths = [], [], []
            dataset = filename.split('.')[0]
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    predictions.append(data["pred"])
                    answers.append(data["answers"])
                    all_classes = data["all_classes"]
                    if "length" in data:
                        lengths.append(data["length"])
            if args.e:
                score = scorer_e(dataset, predictions, answers, lengths, all_classes)
            else:
                score = scorer(dataset, predictions, answers, all_classes)
            scores[dataset] = score
        if args.e:
            out_path = os.path.join(out_path_pred_e, model_name, "result.json")
        else:
            out_path = os.path.join(out_path_pred, model_name, "result.json")
        with open(out_path, "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
            
        print("\n******* Evaluation Results: *******")
        for dataset_name in datasets:
            if dataset_name in scores:
                score = scores[dataset_name]
                print(f"{dataset_name}: {score:.2f}")
                if args.use_wandb:
                    wandb.log({f"{dataset_name}_score": score})
        
if __name__ == "__main__":
    main()

