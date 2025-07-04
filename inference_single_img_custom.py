import os
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPVisionModel, CLIPProcessor, GPT2Tokenizer, GPT2Model
from image_caption_gpt2_clip import ImageCaptionModel  # Your custom decoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def greedy_decode(model, image_tensor, tokenizer, max_len, prompt_tokens, image_model, text_model):
    model.eval()
    input_ids = prompt_tokens.unsqueeze(0).to(DEVICE)
    generated_ids = []

    img_embeddings = image_model(pixel_values=image_tensor.unsqueeze(0).to(DEVICE)).last_hidden_state

    for _ in range(max_len):
        txt_embeddings = text_model(input_ids=input_ids).last_hidden_state
        logits = model(img_embeddings, txt_embeddings)

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        if next_token_id.item() in [tokenizer.eos_token_id, tokenizer.encode(".")[0]]:
            break

        generated_ids.append(next_token_id.item())
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_single_inference(image_path, model_path, max_len=50, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and processor
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Build prompt
    prompt_tokens = tokenizer("Describe this image:", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)

    # Load models
    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE).eval()
    text_model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()
    model = ImageCaptionModel(
        decoder_layers=8,
        attention_heads=4,
        embed_dim=768,
        vocab_size=tokenizer.vocab_size,
        mode="infer",
        device=DEVICE
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully.")

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    image_tensor = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    # Generate caption
    caption = greedy_decode(model, image_tensor, tokenizer, max_len, prompt_tokens, image_model, text_model)

    print("📝 Generated caption:")
    print(f"    {caption}")

    # Save visualization
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
    parser = argparse.ArgumentParser(description="GPT2-CLIP image captioning inference")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained decoder .pth file")
    parser.add_argument("--max-length", type=int, default=90, help="Maximum caption length")
    parser.add_argument("--output-dir", type=str, default="generated_images", help="Directory to save the captioned image")
    args = parser.parse_args()

    run_single_inference(
        image_path=args.image_path,
        model_path=args.model_path,
        max_len=args.max_length,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
    # python gpt2_inference.py --image-path ./images/cat.jpg --model-path ./train_runs/model_20250703_121043/model_20.pth 