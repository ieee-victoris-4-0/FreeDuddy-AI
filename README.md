# 🤖 BuyBuddy AI/ML Module

## 🚀 Project Overview
The **BuyBuddy AI/ML Module** powers the intelligent features of the BuyBuddy marketplace app, connecting local fashion and beauty brands with customers seeking unique clothing, accessories, and more. This repository contains the AI/ML components driving the following key features:
- **Visual Search**: Users can upload a photo to find visually similar products.
- **Text Search**: Users can search by text to find similar product as they imagine
- **Personalized Recommendations**: A tailored homepage with product and seasonal gift suggestions based on user preferences. *(Future Plan)*

This module is designed to be modular, scalable, and easy to integrate with the BuyBuddy backend.

## 📋 Features
- **Image-Based Product Search**: Uses YOLO for object detection and CLIP for image embeddings to match uploaded images with products in the catalog.
- **Text-Based Product Search**: Leverages CLIP's text encoding for natural language queries to find relevant products.
- **API Integration**: Exposes endpoints for image search, text search, and health checks via `model_api.py`.
- **Interactive Testing**: A Streamlit-based interface (`streamlit_app.py`) for testing visual and text search functionalities.
- **Vector Database**: Uses Qdrant for efficient storage and retrieval of image and text embeddings.

## 🛠️ Setup Instructions

To set up the BuyBuddy AI/ML Module locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ieee-victoris-4-0/FreeDuddy-AI.git
   cd FreeDuddy-AI
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Key dependencies include:
   - `torch`: For model inference (CLIP and YOLO).
   - `open-clip-torch`: For CLIP model and tokenizer.
   - `ultralytics`: For YOLO object detection.
   - `qdrant-client`: For vector database operations.
   - `fastapi` and `uvicorn`: For the API server.
   - `streamlit`: For the testing interface.
   

4. **Download Pre-trained Models**
   - The YOLO model (`my_model.pt`) is stored in the `/models` directory. Ensure it is available
   - The CLIP model (`ViT-L-14`, pretrained on `laion2b_s32b_b82k`) is automatically loaded via `open_clip`.

5. **Set Up Environment Variables**
   - Create a `.env` file in the root directory with the following variables:
     ```plaintext
     DATA_URL=https://c9cca5a1-b149-4555-bf54-2d325b2cd2e0.eu-central-1-0.aws.cloud.qdrant.io:6333
     API_KEY=your_qdrant_api_key
     MODEL_PATH=./models
     DATA_PATH=./data
     ```
   - Replace `your_qdrant_api_key` with the actual Qdrant API key. Ensure the `DATA_URL` matches your Qdrant instance.

6. **Verify Qdrant Collection**
   - Ensure the Qdrant collection `Buy-Buddy-VD` exists with the correct vector configuration

## 📁 Folder Structure

| Folder/File            | Description                                      |
|------------------------|--------------------------------------------------|
| `/notebooks`           | Jupyter notebooks for model experimentation and prototyping |
| `/data`                | Sample datasets and preprocessing scripts        |
| `/models`              | Saved models (e.g., `my_model.pt`) and training checkpoints |
| `model_api.py`         | FastAPI backend providing endpoints for image search, text search, and health checks |
| `streamlit_app.py`     | Streamlit script for an interactive interface to test visual and text search |
| `requirements.txt`     | List of Python dependencies |
| `.env`                 | Environment variables for Qdrant and paths |

## 🚀 Usage

### Running the Streamlit App
To test the AI module interactively:
```bash
streamlit run streamlit_app.py
```
- **Access**: Open `http://localhost:8501` in your browser.
- **Features**:
  - Upload an image (JPG, JPEG, PNG) to perform a visual search.
  - Enter a text query to search for products via the chatbot.
  - View results with product images, titles, and prices fetched from the Qdrant database.

### Running the API
To start the backend API:
```bash
python model_api.py
```
To start the FastAPI server:
```bash
uvicorn model_api:app --reload
```
- **Access**: API is available at `http://127.0.0.1:8000/docs`.
- **Endpoints**:
  - `GET /health`: Checks the API and Qdrant connection status.
    ```bash
    http://127.0.0.1:8000/health
    # Response: {"status": "ok"}
    ```
  - `POST /search-image`: Upload an image to find similar products.
  - `POST /search-text`: Submit a text query to find products.
### Training Models
To train or fine-tune models:
1. Navigate to the `/notebooks` directory.
2. Open relevant Jupyter notebooks (e.g., `visual_search_training.ipynb`).
3. Follow the instructions to preprocess data (in `/data`) and train models.
4. Save trained models to the `/models` directory.
5. Update the Qdrant collection with new embeddings if necessary.


## 🧠 Technical Details

### Visual Search
- **Model**: YOLO for object detection, CLIP (`ViT-L-14`, `laion2b_s32b_b82k`) for image embeddings.
- **Preprocessing**:
  - YOLO detects clothing items in the uploaded image and crops bounding boxes.
  - Cropped images are resized to 224x224 and normalized using CLIP's preprocessing pipeline.
- **Vector Search**: CLIP embeddings (768-dimensional) are searched in Qdrant using cosine similarity, returning the top 5 matches.
- **Output**: Product IDs (UUIDs) from the Qdrant collection `Buy-Buddy-VD`.

### Text-Based Search
- **Model**: CLIP (`ViT-L-14`, `laion2b_s32b_b82k`) for text embeddings.
- **Preprocessing**: Text queries are tokenized using CLIP's tokenizer.
- **Vector Search**: Text embeddings (768-dimensional) are searched in Qdrant, returning the top 5 matches.
- **Output**: Product IDs (UUIDs) from the Qdrant collection.

### Qdrant Integration
- **Collection**: `Buy-Buddy-VD` stores product embeddings with payloads (title, price, image URL).
- **Configuration**: 768-dimensional vectors, cosine distance metric.
- **Client**: Connects to a Qdrant Cloud instance (configurable via `.env`).

### Streamlit Interface
- **Purpose**: Provides an interactive UI for testing visual and text search.
- **Functionality**:
  - Image upload for visual search.
  - Text input for chatbot queries.
  - Displays results with product images and details fetched from Qdrant.

### Recommendation Engine
- In Future Plan

## 🛠️ Development and Testing

- **Dependencies**: See `requirements.txt` for a complete list. Key libraries include:
  - `torch` or `tensorflow` for model training
  - `streamlit` for the demo interface
  - `fastapi` or `flask` for the API (update based on your stack)


## 🤝 Contributing

We welcome contributions to improve the BuyBuddy AI/ML Module! To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request with a clear description of your changes.



## 🙏 Acknowledgments
- Thanks to the IEEE Victoris 4.0 team for their contributions.
