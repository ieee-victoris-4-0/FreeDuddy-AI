import streamlit as st
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
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import requests
from io import BytesIO

headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/139.0.0.0 Safari/537.36"
            }

yolo = YOLO("models/my_model.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip, _, prep = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='laion2b_s32b_b82k'
)
clip = clip.to(device)
tokenizer = open_clip.get_tokenizer('ViT-L-14')

client = QdrantClient(
     url="https://c9cca5a1-b149-4555-bf54-2d325b2cd2e0.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.pXjkVzeQm8jGIJ4SYfFpBqPCVAWXdoOc8u-wC2x6dIk",
    timeout=120
)

def fetch_image(url, timeout=30):
    try:
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            return Image.open(BytesIO(r.content))
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to load image from {url}: {e}")
        return None

class Search:
    def __init__(self, clip, prep, tokenizer, yolo, client):
        self.clip = clip
        self.prep = prep
        self.tokenizer = tokenizer
        self.yolo = yolo
        self.client = client

    def __img_pre_pil(self, img_pil):
        ret = []
        img = np.array(img_pil)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = self.yolo.predict(img_cv, verbose=False)
        r = results[0]

        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            crop_pil = Image.fromarray(crop)
            ret.append(crop_pil)

        if not len(ret):
            ret.append(img_pil)
        return ret

    def image_search(self, img_pil):
        clothes = self.__img_pre_pil(img_pil)

        for clothe in clothes:
            query_image = self.prep(clothe).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feat = self.clip.encode_image(query_image)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)

            img_feat = img_feat.detach().cpu().numpy().astype("float32")
            img_feat = img_feat.squeeze().tolist()

            hits = self.client.search(
                collection_name="Buy-Buddy-VD",
                query_vector=img_feat,
                limit=5,
                with_payload=True
            )
            images = []

            for h in hits:
                title = h.payload.get("title")
                price = h.payload.get("price")
                img_url = h.payload.get("image_url")

                if img_url:
                    img = fetch_image(img_url, timeout=30)  # خلي الـ timeout أطول
                    if img:
                        images.append((img, f"{title} - {price}"))


            if images:
                st.image(
                    [r[0] for r in images],
                    caption=[r[1] for r in images],
                    use_column_width=True
                )

    def text_search(self, text):
        query = self.tokenizer([text]).to(device)
        with torch.no_grad():
            text_feat = self.clip.encode_text(query)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.detach().cpu().numpy().astype("float32").squeeze().tolist()

        hits = self.client.search(
            collection_name="Buy-Buddy-VD",
            query_vector=text_feat,
            limit=5,
            with_payload=True
        )

        results = []
        for h in hits:
            title = h.payload.get("title")
            price = h.payload.get("price")
            img_url = h.payload.get("image_url")

            if img_url:
                img = fetch_image(img_url, timeout=30)
                if img:
                    results.append((img, f"{title} - {price}"))


        if results:
            st.image(
                [r[0] for r in results],
                caption=[r[1] for r in results],
                use_column_width=True
            )


# ====== Streamlit App ======
st.set_page_config(page_title="BuyBuddy Search", layout="wide")
st.title("👕 BuyBuddy Visual & Text Search")

# object search
search = Search(clip, prep, tokenizer, yolo, client)

# رفع صورة
uploaded_file = st.file_uploader("📷 ارفع صورة للبحث", type=["jpg", "jpeg", "png"])

# إدخال نص
query_text = st.text_input("✍️ أو اكتب وصف للمنتج")

col1, col2 = st.columns(2)

with col1:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="الصورة المدخلة", use_column_width=True)
        if st.button("🔍 بحث بالصور"):
            search.image_search(image)

with col2:
    if query_text:
        if st.button("🔍 بحث بالنص"):
            search.text_search(query_text)
