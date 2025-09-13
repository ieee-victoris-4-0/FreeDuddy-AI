from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import numpy as np
import pandas as pd
import torch
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, SearchParams
from transformers import CLIPProcessor
import open_clip
import requests
from io import BytesIO
from ultralytics import YOLO  
import cv2
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import uuid
from qdrant_client.http import models
from dotenv import load_dotenv

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/139.0.0.0 Safari/537.36"
}

def fetch_image(url, timeout=30):
    try:
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            return Image.open(BytesIO(r.content))
    except requests.exceptions.RequestException as e:
        print(f"Failed to load image from {url}: {e}")
        return None

app = FastAPI()
# app.mount("/images", StaticFiles(directory="images"), name="images")  # Serve images from filesystem
yolo = YOLO("model.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip, _, prep = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='laion2b_s32b_b82k'   
)

tokenizer = open_clip.get_tokenizer('ViT-L-14')
load_dotenv()
# Initialize Qdrant client
print(f"{os.getenv("API_KEY")}")
print(f"{os.getenv("DATA_URL")}")
qdrant_client = QdrantClient(
    url= os.getenv("DATA_URL"), 
    api_key= os.getenv("API_KEY")# Ensure the API key is set in the environment
)

class Search:
    def __init__(self, clip, prep, tokenizer, yolo, client):
        self.clip = clip
        self.prep = prep
        self.tokenizer = tokenizer
        self.yolo = yolo
        self.client = client

    def __img_pre(self, img_bytes):
        ret = []
        # Convert bytes to OpenCV image
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # YOLO prediction
        results = self.yolo.predict(img, verbose=False)
        r = results[0]
        
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop)
            ret.append(crop_pil)
        
        if not ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            ret.append(img_pil)
        return ret

    def image_search(self, img_bytes):
        clothes = self.__img_pre(img_bytes)
        results = []
        
        # Process only the first detected clothing item for simplicity
        clothe = clothes[0]
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
        # print(f"{hits}")
        for h in hits:
            idx = h.id
            brand = h.payload.get('brand')
            title = h.payload.get('title')
            price = h.payload.get('price')
            imageurl = h.payload.get('image_url')
            # print(f"{imageurl}")
            if imageurl:
                results.append({
                    "id": idx,
                    "brand": brand,
                    "title": title,
                    "price": price,
                    "imageurl": imageurl
                })

        return results

    def text_search(self, text):
        query = [text]
        query = self.tokenizer(query).to(device)
        with torch.no_grad():
            text_feat = self.clip.encode_text(query)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.detach().cpu().numpy().astype("float32")
            text_feat = text_feat.squeeze().tolist()
        
        hits = self.client.search(
            collection_name="Buy-Buddy-VD",
            query_vector=text_feat,
            limit=5,
            with_payload=True
        )
        # print(f"{hits}")
        results = []
        for h in hits:
            idx = h.id
            brand = h.payload.get('brand')
            title = h.payload.get('title')
            price = h.payload.get('price')
            imageurl = h.payload.get('image_url')
            # print(f"{imageurl}")
            if imageurl:
                results.append({
                    "id": idx,
                    "brand": brand,
                    "title": title,
                    "price": price,
                    "imageurl": imageurl
                })

        return results

search = Search(
    clip=clip,
    prep=prep,
    tokenizer=tokenizer,
    yolo=yolo,
    client=qdrant_client
)



@app.post("/search-image")  # Client flow: Image-based search
async def image_search(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()  # Await the file read to get bytes
        print(f"{image_bytes}")
        if not isinstance(image_bytes, bytes):
            return {"status": "error", "message": "Invalid file content"}
        results = search.image_search(image_bytes)
        print(f"{results}")
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/search-text")  # Client flow: Text-based search
async def text_search(query: str = Form(...)):
    try:
        results = search.text_search(query)
        print(f"{results}")
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    



