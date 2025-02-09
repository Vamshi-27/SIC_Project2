import torch
from PIL import Image

def print_progress(epoch, batch_idx, total_batches, loss):
    print(f"[Epoch {epoch}] Batch {batch_idx}/{total_batches} - Loss: {loss:.4f}")

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def tokenize_text(text, tokenizer):
    return tokenizer.encode(text, return_tensors="pt")
