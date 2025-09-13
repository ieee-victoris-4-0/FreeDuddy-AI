# 🤖 BuyBuddy AI/ML Module

## 🚀 Project Overview
The **BuyBuddy AI/ML Module** powers the intelligent features of the BuyBuddy marketplace app, connecting local fashion and beauty brands with customers seeking unique clothing, accessories, and more. This repository contains the AI/ML components driving the following key features:
- **Visual Search**: Users can upload a photo to find visually similar products.
- **Text Search**: Users can search by text to find similar product as they imagine
- **Personalized Recommendations**: A tailored homepage with product and seasonal gift suggestions based on user preferences. *(Future Plan)*

This module is designed to be modular, scalable, and easy to integrate with the BuyBuddy backend.

## 📋 Features
- **Image-Based Product Search**: Uses computer vision to match uploaded images with products in the database.
- **Text-Based Product Search**: A textual search to match the query with products in the database.
- **Recommendation Engine**: Leverages user data to provide personalized product and gift recommendations. *(Future-Plan)*
- **API Integration**: Exposes endpoints for image search, text search, and health checks via `model_api.py`.

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
   ```


4. **Set Up Environment Variables**
   - Create a `.env` file in the root directory with the following variables:
     ```plaintext
     API_KEY=your_api_key_here
     DATA_URL=your_url
     MODEL_PATH=./models
     ```
   - Update paths or keys as needed based on your setup, as the database is uploaded on qdrant server.

## 📁 Folder Structure

| Folder/File            | Description                                      |
|------------------------|--------------------------------------------------|
| `/notebooks`           | Jupyter notebooks for model experimentation and prototyping |
| `/data`                | Sample datasets and preprocessing scripts        |
| `/models`              | Saved models and training checkpoints            |
| `model_api.py`         | Backend API file providing endpoints for image search, text search, and health checks |
| `streamlit_app.py`     | Script to run a Streamlit-based interface for testing the AI module |

## 🚀 Usage

### Running the Streamlit App
To test the AI module locally using the Streamlit interface:
```bash
streamlit run streamlit_app.py
```
This will launch a web interface at `http://localhost:8501` where you can test the visual search, chatbot, and recommendation features.

### Running the API
To start the backend API:
```bash
python model_api.py
```
Available endpoints:
- `POST /image-search`: Upload an image to find similar products.
- `POST /text-search`: Query the chatbot with text input.
- `GET /health`: Check the API's health status.

Example API call (using `uvicorn`):
```bash
uvicorn model_api:app --reload
```
- Then take the hostlink (usually: `http://127.0.0.1:8000`)
- Add `/docs` in the end of hostlink to view Swagger UI of API checkpoints
### Training Models
To train or fine-tune models:
1. Navigate to the `/notebooks` directory.
2. Open the relevant Jupyter notebook (e.g., `deepfashion-yolo.ipynb`).
3. Save trained models to the `/models` directory.

## 🧠 Technical Details

### Visual Search
- **Model**: [YOLO, CLIP]
- **Framework**: PyTorch
- **Process**: Convert images into embeddings and save them in vector database on qdrant database and upload on qdrant server (Note: take your api token and url and put them in `.env` file)
- **Output**: Returns a list of product IDs with similarity scores.

### Text Search
- **Model**: CLIP
- **Framework**: Transformers in open-clip
- **Input**: Text queries from users.
- **Output**: Relevant product matches

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
