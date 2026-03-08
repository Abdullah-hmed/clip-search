import clip, torch, os
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP model...")

model, preprocess = clip.load("ViT-L/14", device=device)

text = "a photo of a woman with visible nipples"
text_emb = model.encode_text(clip.tokenize(text).to(device))
text_emb /= text_emb.norm(dim=-1, keepdim=True)

folder = "C:/Users/abdul/Downloads/Comfy-assisted-dataset"

print("Searching Images...")

for f in os.listdir(folder):
    if not f.lower().endswith((".jpg",".png",".jpeg",".webp")):
        continue

    img = preprocess(Image.open(os.path.join(folder, f))).unsqueeze(0).to(device)
    img_emb = model.encode_image(img)
    img_emb /= img_emb.norm(dim=-1, keepdim=True)

    sim = (text_emb @ img_emb.T).item()
    print(f, sim)
