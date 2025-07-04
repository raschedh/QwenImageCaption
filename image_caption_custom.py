# we need to encode an image
# we need to encode text
# then we need to send the concat with a start token through a decoder that we train
# the decoder can be qwen but to start we can use the transformer decoder
# at test time we take an image and describe the image 
import torch 
import torch.nn as nn
from torch import Tensor 
from torch.nn.functional import cosine_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, CLIPProcessor, GPT2Tokenizer, GPT2Model
from dataloader import Flickr30KDataloader, collate_fn
from datasets import load_from_disk
from datetime import datetime
from tqdm import tqdm
import os 

# ---------- MultiHead Attention Class -------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int):
        super().__init__()

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        assert self.embed_dim % self.n_heads == 0, "embed_dim must be divisible by n_heads"

        self.D_per_head = self.embed_dim // self.n_heads
        self.linear_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask = None):    
        
        B, N_q, _ = queries.shape
        N_k = keys.shape[1]
        N_v = values.shape[1] 

        Q = self.linear_q(queries)
        K = self.linear_k(keys)
        V = self.linear_v(values)

        # reshape them from (B, N_{q,k,v}, D) to (B, N_{q,k,v}, heads, Dh) and then permute to (B, heads, N_{q,k,v}, Dh)
        Q = Q.reshape(B, N_q, self.n_heads, self.D_per_head).permute(0,2,1,3)
        K = K.reshape(B, N_k, self.n_heads, self.D_per_head).permute(0,2,1,3)
        V = V.reshape(B, N_v, self.n_heads, self.D_per_head).permute(0,2,1,3)
                                
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.D_per_head ** 0.5)

        if mask is not None:
            # mask = torch.triu(torch.ones(N_q, N_q, device=queries.device), diagonal=1)
            # mask = mask.masked_fill(mask == 1, float("-inf"))
            # mask = mask.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, N, N)
            attention_scores = attention_scores + mask # torch takes care of broadcasting to (B, heads, N, N)
            
        attn_weights = torch.softmax(attention_scores, dim=-1)

        attention = torch.matmul(attn_weights, V)
        attention = attention.permute(0,2,1,3)
        attention = attention.reshape(B, N_q, self.embed_dim)
        attention = self.linear_layer(attention)
        return attention

# ---------- Decoder Class (without cross attention) -------------
class Decoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 attention_heads: int,
                 scale: int):

        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.masked_attention = MultiHeadAttention(attention_heads, embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, scale * embed_dim),
            nn.GELU(),
            nn.Linear(scale * embed_dim, embed_dim)
        )

    def forward(self, x: Tensor, mask: Tensor):
        x = self.masked_attention(x,x,x, mask) + x
        x = self.layer_norm1(x)
        x = self.feed_forward(x) + x
        x = self.layer_norm2(x)
        return x
    
# ----------------- POSITIONAL EMBEDDING CLASS -------------------------------
class PositionEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int):

        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        
        positions = torch.arange(0, max_len, step=1).float().unsqueeze(1)
        power = 10000 ** (torch.arange(0, embed_dim, step=2) / embed_dim) # this is the 10000^(2i/D) part in the orignal paper
        
        pe[:, 0::2] = torch.sin(positions / power)
        pe[:, 1::2] = torch.cos(positions / power)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        return x + self.pe[:x.size(1), :]

# ---------- Final Multimodal Class ------------- 
class ImageCaptionModel(nn.Module):
    def __init__(self, 
                decoder_layers, 
                attention_heads, 
                embed_dim, 
                vocab_size,
                mode,
                device):
        super().__init__()

        self.mode = mode 
        self.device = device
        self.embed_dim = embed_dim
        self.pos_embed = PositionEmbedding(embed_dim=768, max_len=500)
        self.decoder_layers = nn.ModuleList([
            Decoder(embed_dim, attention_heads, scale=4)
            for _ in range(decoder_layers)
        ])
        self.final_linear = nn.Linear(embed_dim, vocab_size)  # GPT vocab size

        if self.mode == "train":
            image_tokens, text_tokens = img_embeddings.size(1), txt_embeddings.size(1)
            N = image_tokens + text_tokens
            mask = torch.zeros((N, N), device=self.device)
            # Mask future tokens in the text-text attention block
            causal_block = torch.triu(torch.ones((text_tokens, text_tokens), device=self.device), diagonal=1)
            causal_block = causal_block.masked_fill(causal_block == 1, float('-inf'))
            mask[image_tokens:, image_tokens:] = causal_block # Put this in the lower-right block of the full mask
            mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, N, N] to match attention broadcast
        elif self.mode == "infer":
            mask = None
        else:
            raise ValueError(f"Invalid mode must be 'train' or 'infer' but given: {self.mode}")

    def forward(self, img_embeddings, txt_embeddings):
        
        x = torch.cat([img_embeddings, txt_embeddings], dim=1)
        x = self.pos_embed(x)

        for layer in self.decoder_layers:
            x = layer(x, self.mask)

        x = x[:, -txt_embeddings.size(1):, :]  # obtain just the text
        logits = self.final_linear(x)  # map to vocab
        return logits
    

if __name__ == "__main__":
    EPOCHS = 30
    BATCH_SIZE = 32 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_from_disk("./flickr_dataset/train") # path to flickr data
    val_data = load_from_disk("./flickr_dataset/val") # path to flickr data

    # Load CLIP model and processor
    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE)
    image_model = image_model.eval()
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load GPT-2 tokenizer and model
    text_model = GPT2Model.from_pretrained("gpt2").to(DEVICE)
    text_model = text_model.eval()
    text_tokenisor = GPT2Tokenizer.from_pretrained("gpt2")

    PAD_TOKEN_ID = text_tokenizer.pad_token_id or text_tokenizer.eos_token_id # GPT2 has no explicit pad token or start
    VOCAB_SIZE = text_tokenisor.vocab_size

    train_dataset = Flickr30KDataloader(train_data, tokenizer=text_tokenisor, processor=image_processor, max_length=91)
    val_dataset = Flickr30KDataloader(val_data, tokenizer=text_tokenisor, processor=image_processor, max_length=91)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = ImageCaptionModel(decoder_layers = 8, 
                             attention_heads = 4, 
                             embed_dim = 768, 
                             vocab_size = VOCAB_SIZE,
                             mode = "train",
                             device = DEVICE).to(DEVICE)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    best_val_loss = float("inf")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(f"train_runs/run_{timestamp}")
    model_dir = os.path.join(f"train_runs/model_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        for images, captions, attn_mask in tqdm(train_loader, desc="Training"):
            
            images, captions, attn_mask = images.to(DEVICE), captions.to(DEVICE), attn_mask.to(DEVICE)

            with torch.no_grad():
                img_embeddings = image_model(pixel_values=images).last_hidden_state
                txt_embeddings = text_model(input_ids=captions, attention_mask=attn_mask).last_hidden_state # [B, T, 768]
                
            input_emb = txt_embeddings[:, :-1]

            logits = model(img_embeddings, input_emb)  # [B, T-1, vocab_size]
            targets = captions[:, 1:]  # [B, T-1], token IDs you want the model to predict

            logits_flat = logits.reshape(-1, logits.size(-1))     # [B*T, vocab_size]
            targets_flat = targets.reshape(-1)                    # [B*T]

            loss = criterion(logits_flat, targets_flat)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, captions, attn_mask in tqdm(val_loader, desc="Validating"):

                images, captions, attn_mask = images.to(DEVICE), captions.to(DEVICE), attn_mask.to(DEVICE)

                img_embeddings = image_model(pixel_values=images).last_hidden_state
                txt_embeddings = text_model(input_ids=captions, attention_mask=attn_mask).last_hidden_state # [B, T, 768]

                input_emb = txt_embeddings[:, :-1]

                logits = model(img_embeddings, input_emb)  # [B, T-1, vocab_size]
                targets = captions[:, 1:]  # [B, T-1], token IDs you want the model to predict

                logits_flat = logits.reshape(-1, logits.size(-1))     # [B*T, vocab_size]
                targets_flat = targets.reshape(-1)                    # [B*T]

                loss = criterion(logits_flat, targets_flat)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, f"model_{epoch}.pth"))
            print(f"✅ Saved better model")
    
    writer.close()
