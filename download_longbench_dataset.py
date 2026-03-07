#!/usr/bin/env python3
"""
script to download the LongBench datasets
"""

import os
from datasets import load_dataset
import json

def download_longbench_dataset():
    """download LongBench datasets"""
    
    # set environment variables
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # LongBench dataset list
    datasets = [
        "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", 
        "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", 
        "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", 
        "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", 
        "lcc", "repobench-p"
    ]
    
    # LongBench-E dataset list
    datasets_e = [
        "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", 
        "multi_news", "trec", "triviaqa", "samsum", "passage_count", 
        "passage_retrieval_en", "lcc", "repobench-p"
    ]
    
    print("Start downloading LongBench datasets...")
    
    # create data save directory
    save_dir = "./longbench_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # download all datasets
    for dataset_name in datasets:
        print(f"Downloading dataset: {dataset_name}")
        try:
            # download test split
            dataset = load_dataset(
                "/dataset/tao/longbench/THUDM___long_bench",
                "default",
                split="test",
                trust_remote_code=True,
            )
            
            # # save locally
            # dataset_path = os.path.join(save_dir, f"{dataset_name}_test")
            # dataset.save_to_disk(dataset_path)
            # print(f"✓ {dataset_name} test split downloaded, saved to: {dataset_path}")
            
            # print dataset information
            print(f"  - dataset size: {len(dataset)} samples")
            if len(dataset) > 0:
                print(f"  - sample fields: {list(dataset[0].keys())}")
            
        except Exception as e:
            print(f"✗ download {dataset_name} failed: {str(e)}")
    
    print("\nStart downloading LongBench-E datasets...")
    
    # download LongBench-E datasets
    for dataset_name in datasets_e:
        print(f"Downloading LongBench-E dataset: {dataset_name}")
        try:
            # download test split
            dataset = load_dataset(
                "/dataset/tao/longbench/THUDM___long_bench",
                "default",
                split="test",
                trust_remote_code=True,
            )
            
            # # save locally
            # dataset_path = os.path.join(save_dir, f"{dataset_name}_e_test")
            # dataset.save_to_disk(dataset_path)
            # print(f"✓ {dataset_name} LongBench-E test split downloaded, saved to: {dataset_path}")
            
            # print dataset information
            print(f"  - dataset size: {len(dataset)} samples")
            if len(dataset) > 0:
                print(f"  - sample fields: {list(dataset[0].keys())}")
            
        except Exception as e:
            print(f"✗ download {dataset_name} LongBench-E failed: {str(e)}")
    
    # print(f"\nAll datasets downloaded! Data saved at: {save_dir}")
    
    # # create dataset info file
    # info = {
    #     "datasets": datasets,
    #     "datasets_e": datasets_e,
    #     "save_dir": save_dir,
    #     "total_datasets": len(datasets) + len(datasets_e)
    # }
    
    # with open(os.path.join(save_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
    #     json.dump(info, f, indent=2, ensure_ascii=False)
    
    # print(f"dataset info saved to: {os.path.join(save_dir, 'dataset_info.json')}")

def download_single_dataset(dataset_name):
    """download a single dataset"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    save_dir = "./longbench_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Downloading dataset: {dataset_name}")
    try:
        dataset = load_dataset(
            "THUDM/LongBench",
            dataset_name,
            split="test",
            trust_remote_code=True
        )
        
        dataset_path = os.path.join(save_dir, f"{dataset_name}_test")
        dataset.save_to_disk(dataset_path)
        print(f"✓ {dataset_name} downloaded, saved to: {dataset_path}")
        print(f"  - dataset size: {len(dataset)} samples")
        if len(dataset) > 0:
            print(f"  - sample fields: {list(dataset[0].keys())}")
            
    except Exception as e:
        print(f"✗ download {dataset_name} failed: {str(e)}")

if __name__ == "__main__":
    # import argparse
    
    # parser = argparse.ArgumentParser(description="download LongBench datasets")
    # parser.add_argument("--dataset", type=str, default=None, 
    #                    help="specify a single dataset to download; if omitted, download all datasets")
    
    # args = parser.parse_args()
    
    # if args.dataset:
    #     download_single_dataset(args.dataset)
    # else:
    download_longbench_dataset() 