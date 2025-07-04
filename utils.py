from datasets import load_dataset, DatasetDict, load_from_disk
import os
from collections import defaultdict
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2Model, CLIPProcessor, CLIPVisionModel
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def getFlickrData(save_dir, dataset_path):

    print("Loading Flickr Data from Hugging Face...")

    os.makedirs(save_dir, exist_ok=True)
    ds = load_dataset(dataset_path)["test"]

    print(f"Saving dataset to {save_dir}...")
    # Convert the single split into multiple Dataset objects
    train_dataset = ds.filter(lambda x: x["split"] == "train")
    val_dataset = ds.filter(lambda x: x["split"] == "val")
    test_dataset = ds.filter(lambda x: x["split"] == "test")

    # Organize into proper split dict
    dataset = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    })

    dataset.save_to_disk(save_dir)
    print("✅ All splits saved successfully.")
    return     


def getQwenModel(save_path):

    os.makedirs(save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f"✅ Qwen3 model and tokenizer saved to: {save_path}")
    return 

if __name__ == "__main__":
    getQwenModel("./qwen3model")
    getFlickrData(save_dir = "flickr_dataset", dataset_path="nlphuji/flickr30k") 