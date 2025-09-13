import numpy as np 
import pandas as pd
from qdrant_client.http import models
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import os

vecs = np.load("embeddings\image_vecs.npy")
df = pd.read_csv("data/final_dataset.csv")

ids = df['index'].to_list()
titless = df['title'].to_list()
paths = df['local_image_path'].to_list()
prices = df['full_price'].to_list()
brands = df['brand'].to_list()
img_urls = df['imageurl'].to_list()
itm_urls = df['itemurl'].to_list()

collection_name = "Buy-Buddy-VD"

client = QdrantClient(
    url= os.getenv("DATA_URL"), 
    api_key= os.getenv("API_KEY")# Ensure the API key is set in the environment
)

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

batch_size = 100
total = len(vecs)

for start in tqdm(range(0, total, batch_size), desc="Upserting with payloads"):
    end = min(start + batch_size, total)

    points = [
        models.PointStruct(
            id=ids[i],
            vector=vecs[i],  # لو انت مش محتاج تعدل الـ vector حط نفس القديم
            payload={
                "title": titless[i],
                "path": paths[i],
                "price": prices[i],
                "brand": brands[i],
                "image_url": img_urls[i],
                "item_url": itm_urls[i]
            }
        )
        for i in range(start, end)
    ]

    client.upsert(
        collection_name=collection_name,
        points=points
    )

print("✅ Done! Upserted:", total, "points with payloads")