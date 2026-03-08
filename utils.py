import clip
import torch

def encode_text(model, text, device):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
    return text_emb


def encode_image(model, image, device):
    with torch.no_grad():
        img_emb = model.encode_image(image.to(device))
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
    return img_emb

def compute_similarity(text_emb, img_emb):
    return (text_emb @ img_emb.T).item()  # cosine similarity

def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))

def load_model(device):
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess