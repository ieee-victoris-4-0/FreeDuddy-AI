# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
import cv2
import os
import torch
import open_clip
from PIL import Image
import faiss
from tqdm import tqdm

# load dataset
df = pd.read_csv("data/dataset.csv")

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ViT-L-14"
pretrained = "laion2b_s32b_b82k"

model, preprocess, _ = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained
)

model = model.to(device)
model.eval()

# function to embedd images
def get_image_embedding(image_path):
    img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
    emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

# initialize faiss files with indecies
dim = 768
image_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
image_vecs = []

# Convert all images
for _, row in tqdm(df.iterrows(), total=len(df)):
    id = row["index"]
    image_emb = get_image_embedding(row["local_image_path"])[0]

    # add image embedding
    image_index.add_with_ids(
        np.array(image_emb, dtype="float32").reshape(1, -1),
        np.array([id], dtype="int64")
    )

    image_vecs.append(image_emb)

# Save files
faiss.write_index(image_index, "embeddings/image_index.faiss")
np.save("embeddings/image_vecs.npy", np.array(image_vecs))