import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from dataloader import Flickr30KDataloader, collate_fn
from datasets import load_from_disk
from datetime import datetime
from tqdm import tqdm
import os

# ---------------------- Qwen Wrapper with Image Prefix ----------------------
class QwenWithImagePrefix(nn.Module):
    def __init__(self, qwen_model, image_embed_dim, qwen_embed_dim):
        super().__init__()
        self.qwen = qwen_model
        self.img_proj = nn.Linear(image_embed_dim, qwen_embed_dim)

    def forward(self, image_embeds, input_ids, attention_mask):
        prefix_len = image_embeds.size(1)  # dynamically get the number of image tokens

        img_tokens = self.img_proj(image_embeds)  # [B, prefix_len, 1024], upscale image to match qwen hidden dimensions
        tok_embeddings = self.qwen.model.embed_tokens(input_ids)  # [B, T, 1024]
        full_embeddings = torch.cat([img_tokens, tok_embeddings], dim=1)  # [B, prefix + T, 1024]

        prefix_mask = torch.ones((input_ids.size(0), prefix_len), device=input_ids.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, prefix_len + T]

        outputs = self.qwen(inputs_embeds=full_embeddings, attention_mask=full_mask)

        return outputs.logits[:, prefix_len:, :]  # [B, T, V] → skip image token outputs

# ---------------------- Main Training Script ----------------------
if __name__ == "__main__":
    EPOCHS = 30
    BATCH_SIZE = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    train_data = load_from_disk("./flickr_dataset/train") # path to flickr data, see utils.py to generate flickr data from huggingface
    val_data = load_from_disk("./flickr_dataset/val")

    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE).eval()
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_model.eval()

    text_tokenizer = AutoTokenizer.from_pretrained("./qwen3", use_fast=True)
    text_model = AutoModelForCausalLM.from_pretrained("./qwen3").to(DEVICE)
    text_model.train()

    PAD_TOKEN_ID = text_tokenizer.pad_token_id or text_tokenizer.eos_token_id

    train_dataset = Flickr30KDataloader(train_data, tokenizer=text_tokenizer, processor=image_processor, max_length=91)
    val_dataset = Flickr30KDataloader(val_data, tokenizer=text_tokenizer, processor=image_processor, max_length=91)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = QwenWithImagePrefix(qwen_model=text_model, image_embed_dim=768, qwen_embed_dim=1024).to(DEVICE)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("qwen_runs", f"run_{timestamp}")
    model_dir = os.path.join("qwen_runs", f"model_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        for images, captions, attn_mask in tqdm(train_loader, desc="Training"):
            images, captions, attn_mask = images.to(DEVICE), captions.to(DEVICE), attn_mask.to(DEVICE)

            with torch.no_grad():
                img_embeddings = image_model(pixel_values=images).last_hidden_state  # [B, 50, 768]

            input_ids, targets = captions[:, :-1], captions[:, 1:]
            attention_mask = attn_mask[:, :-1]

            logits = model(img_embeddings, input_ids, attention_mask)  # already sliced
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, captions, attn_mask in tqdm(val_loader, desc="Validating"):
                images, captions, attn_mask = images.to(DEVICE), captions.to(DEVICE), attn_mask.to(DEVICE)
                img_embeddings = image_model(pixel_values=images).last_hidden_state

                input_ids, targets = captions[:, :-1], captions[:, 1:]
                attention_mask = attn_mask[:, :-1]

                logits = model(img_embeddings, input_ids, attention_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, f"model_{epoch}.pth"))
            print(f"✅ Saved better model")

    writer.close()
