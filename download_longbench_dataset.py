#!/usr/bin/env python3
"""
下载 LongBench 数据集的脚本
"""

import os
from datasets import load_dataset
import json

def download_longbench_dataset():
    """下载 LongBench 数据集"""
    
    # 设置环境变量
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # LongBench 数据集列表
    datasets = [
        "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", 
        "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", 
        "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", 
        "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", 
        "lcc", "repobench-p"
    ]
    
    # LongBench-E 数据集列表
    datasets_e = [
        "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", 
        "multi_news", "trec", "triviaqa", "samsum", "passage_count", 
        "passage_retrieval_en", "lcc", "repobench-p"
    ]
    
    print("开始下载 LongBench 数据集...")
    
    # 创建数据保存目录
    save_dir = "./longbench_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 下载所有数据集
    for dataset_name in datasets:
        print(f"正在下载数据集: {dataset_name}")
        try:
            # 下载测试集
            dataset = load_dataset(
                "/dataset/tao/longbench/THUDM___long_bench",
                "default",
                split="test",
                trust_remote_code=True,
            )
            
            # # 保存到本地
            # dataset_path = os.path.join(save_dir, f"{dataset_name}_test")
            # dataset.save_to_disk(dataset_path)
            # print(f"✓ {dataset_name} 测试集下载完成，保存到: {dataset_path}")
            
            # 打印数据集信息
            print(f"  - 数据集大小: {len(dataset)} 个样本")
            if len(dataset) > 0:
                print(f"  - 样本字段: {list(dataset[0].keys())}")
            
        except Exception as e:
            print(f"✗ 下载 {dataset_name} 失败: {str(e)}")
    
    print("\n开始下载 LongBench-E 数据集...")
    
    # 下载 LongBench-E 数据集
    for dataset_name in datasets_e:
        print(f"正在下载 LongBench-E 数据集: {dataset_name}")
        try:
            # 下载测试集
            dataset = load_dataset(
                "/dataset/tao/longbench/THUDM___long_bench",
                "default",
                split="test",
                trust_remote_code=True,
            )
            
            # # 保存到本地
            # dataset_path = os.path.join(save_dir, f"{dataset_name}_e_test")
            # dataset.save_to_disk(dataset_path)
            # print(f"✓ {dataset_name} LongBench-E 测试集下载完成，保存到: {dataset_path}")
            
            # 打印数据集信息
            print(f"  - 数据集大小: {len(dataset)} 个样本")
            if len(dataset) > 0:
                print(f"  - 样本字段: {list(dataset[0].keys())}")
            
        except Exception as e:
            print(f"✗ 下载 {dataset_name} LongBench-E 失败: {str(e)}")
    
    # print(f"\n所有数据集下载完成！数据保存在: {save_dir}")
    
    # # 创建数据集信息文件
    # info = {
    #     "datasets": datasets,
    #     "datasets_e": datasets_e,
    #     "save_dir": save_dir,
    #     "total_datasets": len(datasets) + len(datasets_e)
    # }
    
    # with open(os.path.join(save_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
    #     json.dump(info, f, indent=2, ensure_ascii=False)
    
    # print(f"数据集信息已保存到: {os.path.join(save_dir, 'dataset_info.json')}")

def download_single_dataset(dataset_name):
    """下载单个数据集"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    save_dir = "./longbench_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"正在下载数据集: {dataset_name}")
    try:
        dataset = load_dataset(
            "THUDM/LongBench",
            dataset_name,
            split="test",
            trust_remote_code=True
        )
        
        dataset_path = os.path.join(save_dir, f"{dataset_name}_test")
        dataset.save_to_disk(dataset_path)
        print(f"✓ {dataset_name} 下载完成，保存到: {dataset_path}")
        print(f"  - 数据集大小: {len(dataset)} 个样本")
        if len(dataset) > 0:
            print(f"  - 样本字段: {list(dataset[0].keys())}")
            
    except Exception as e:
        print(f"✗ 下载 {dataset_name} 失败: {str(e)}")

if __name__ == "__main__":
    # import argparse
    
    # parser = argparse.ArgumentParser(description="下载 LongBench 数据集")
    # parser.add_argument("--dataset", type=str, default=None, 
    #                    help="指定要下载的单个数据集名称，如果不指定则下载所有数据集")
    
    # args = parser.parse_args()
    
    # if args.dataset:
    #     download_single_dataset(args.dataset)
    # else:
    download_longbench_dataset() 