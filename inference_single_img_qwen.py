import torch
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from image_caption_qwen_clip import QwenWithImagePrefix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def greedy_decode(model, image_tensor, tokenizer, max_len, prompt_tokens, image_model):
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


def run_single_inference(image_path, model_path, tokenizer_path, max_len=50, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Prompt tokens
    prompt_tokens = tokenizer("Describe this image:", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)

    # Load models
    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE).eval()
    qwen = AutoModelForCausalLM.from_pretrained(tokenizer_path).to(DEVICE).eval()
    model = QwenWithImagePrefix(qwen_model=qwen, image_embed_dim=768, qwen_embed_dim=1024).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully.")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    # Generate caption
    caption = greedy_decode(model, image_tensor, tokenizer, max_len, prompt_tokens, image_model)

    print("📝 Generated caption:")
    print(f"    {caption}")

    # Save captioned image
    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{filename}_captioned.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.set_title("Generated Caption", fontsize=12, fontweight='bold')
    ax.text(0.5, -0.15, caption,
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📁 Saved captioned image to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen-CLIP single image inference")
    parser.add_argument("--image-path", type=str, required=True, help="Path to image file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--tokenizer-path", type=str, default="./qwen3model", help="Path to tokenizer/model (e.g. ./qwen3model)")
    parser.add_argument("--max-length", type=int, default=91, help="Max length of generated caption")
    parser.add_argument("--output-dir", type=str, default="generated_images", help="Where to save the output image")
    args = parser.parse_args()

    run_single_inference(
        image_path=args.image_path,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        max_len=args.max_length,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
    # python inference_single_img_qwen.py --image-path ./images/dog.jpg --model-path ./qwen_img_weights.pth