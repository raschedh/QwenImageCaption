#!/usr/bin/env python3

import torch
import argparse
import os
import random
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import CLIPProcessor, AutoTokenizer, CLIPVisionModel, AutoModelForCausalLM
from PIL import Image
from image_caption_qwen_clip import QwenWithImagePrefix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def generate_caption(model, image_tensor, tokenizer, max_len, prompt_tokens, image_model):
    model.eval()
    input_ids = prompt_tokens.unsqueeze(0).to(DEVICE)
    attention_mask = torch.ones_like(input_ids)

    img_embeddings = image_model(pixel_values=image_tensor.unsqueeze(0).to(DEVICE)).last_hidden_state
    generated_ids = []

    for _ in range(max_len):
        logits = model(img_embeddings, input_ids=input_ids, attention_mask=attention_mask)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_id.item())
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id.unsqueeze(0))], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def test_model_on_random_samples(model, dataset, tokenizer, processor, image_model,
                                 num_samples=3, max_length=50, output_dir="generated_images"):
    dataset_size = len(dataset)
    print(f"📁 Dataset size: {dataset_size}")

    if dataset_size == 0:
        print("❌ No samples found in the test dataset.")
        return

    os.makedirs(output_dir, exist_ok=True)

    num_samples = min(num_samples, dataset_size)
    random_indices = random.sample(range(dataset_size), num_samples)
    selected_samples = [dataset[i] for i in random_indices]

    prompt_tokens = tokenizer("Describe this image:", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)

    for i, sample in enumerate(selected_samples):
        image = sample["image"].convert("RGB")
        reference_caption = random.choice(sample["caption"])
        image_tensor = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        generated_caption = generate_caption(model, image_tensor, tokenizer, max_length, prompt_tokens, image_model)

        print(f"\n📸 Sample {i + 1}")
        print(f"   Generated: {generated_caption}")
        print(f"   Reference: {reference_caption}")

        # Create and save individual figure
        fig_ind, ax_ind = plt.subplots(figsize=(5, 5))
        ax_ind.imshow(image)
        ax_ind.set_title(f"Sample {i + 1}", fontsize=12, fontweight='bold')
        ax_ind.text(0.5, -0.15,
                    f"Generated: {generated_caption}\nReference: {reference_caption}",
                    transform=ax_ind.transAxes,
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_ind.axis('off')

        image_path = os.path.join(output_dir, f"sample_{i + 1}.png")
        fig_ind.tight_layout()
        fig_ind.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close(fig_ind)


def main():
    parser = argparse.ArgumentParser(description="Test Qwen-CLIP image captioning model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .pth")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to local Flickr30k dataset split (e.g. ./flickr_dataset/test)")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to test")
    parser.add_argument("--max-length", type=int, default=91, help="Max caption generation length")
    args = parser.parse_args()

    # Load tokenizer and vision processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("./qwen3model", use_fast=True)

    # Load models
    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE).eval()
    qwen = AutoModelForCausalLM.from_pretrained("./qwen3model").to(DEVICE).eval()
    model = QwenWithImagePrefix(qwen_model=qwen, image_embed_dim=768, qwen_embed_dim=1024).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully!")

    # Load test dataset
    print(f"📁 Loading dataset from: {args.data_dir}")
    test_data = load_from_disk(args.data_dir)

    # Run test
    test_model_on_random_samples(
        model=model,
        dataset=test_data,
        tokenizer=tokenizer,
        processor=processor,
        image_model=image_model,
        num_samples=args.num_samples,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
    # python test_model_qwen_random.py --model-path qwen_img_weights.pth --data-dir ./flickr_dataset/test --num-samples 4 --plot-path test_results.png