# Image Captioning with CLIP + Qwen 3 & Custom Transformer Decoders

This repo contains code for training and evaluating image captioning models that integrate CLIP's vision encoder with two distinct decoder strategies:
- A **Qwen 3 (0.6B)** language model augmented with vision input.
- A **custom Transformer decoder** trained from scratch.

Both models are trained on the **Flickr30k dataset** to generate natural language descriptions for images.

---

## 📦 Project Structure

```bash
.
├── inference_single_img_qwen.py         # Inference with Qwen 3 decoder on a single image
├── inference_single_img_custom.py       # Inference with custom decoder on a single image
├── inference_flickrtest_qwen.py         # Evaluate Qwen model on Flickr30k test split
├── image_caption_qwen.py                # Training script for Qwen decoder
├── image_caption_custom.py              # Training script for custom decoder
├── utils.py                             # Helper functions (e.g., Flickr30k loader)
├── flickr_dataset/                      # Locally saved train/val/test splits (via `load_from_disk`)
└── qwen3model/                          # Directory containing the base Qwen model
