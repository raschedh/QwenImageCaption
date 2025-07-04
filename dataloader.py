from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random 
import torch

def collate_fn(batch):
    images, input_ids, attention_masks = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(input_ids),
        torch.stack(attention_masks)
    )

class Flickr30KDataloader(Dataset):
    def __init__(self, dataset, tokenizer, processor, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length 
        # self.start_token = tokenizer.bos_token_id
        self.end_token = tokenizer.eos_token_id
        self.start_prompt = tokenizer("Describe this image:", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)

        if self.tokenizer.pad_token_id is None:
            self.pad_token = tokenizer.eos_token_id
        else:
            self.pad_token = tokenizer.pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]
        caption = random.choice(sample["caption"])
        image = sample["image"].convert("RGB")

        tokens = self.tokenizer(caption,
                                padding=False,  # Do not apply padding here, we will handle it later
                                truncation=True,
                                return_tensors="pt",
                                add_special_tokens=False  # Don't automatically add special tokens
                                )

        # We add custom start token for GPT2 and Qwen tokenisors as they have no explicit ones
        input_ids = tokens["input_ids"].squeeze(0)  # [T] (Tokenized caption without special tokens)

        if input_ids.size(0) > self.max_length - self.start_prompt.size(0) - 1:
            input_ids = input_ids[:self.max_length - self.start_prompt.size(0) - 1]

        input_ids = torch.cat([
            self.start_prompt.to(input_ids.device),
            input_ids,
            torch.tensor([self.end_token], device=input_ids.device)
        ])

        original_len = input_ids.size(0)
        padding_length = self.max_length - original_len
        attention_mask = torch.ones(self.max_length, dtype=torch.long, device=input_ids.device)

        if padding_length > 0:
            pad_tensor = torch.tensor([self.pad_token] * padding_length, device=input_ids.device)
            input_ids = torch.cat([input_ids, pad_tensor])
            attention_mask[original_len:] = 0 

        image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)  # [3, H, W]

        return image_tensor, input_ids, attention_mask
