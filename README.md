# MulberryAI — Mulberry Leaf Disease Detector

AI-powered web app for detecting mulberry leaf diseases using MobileNetV2 transfer learning.

## Setup

### Backend (Flask + TensorFlow)
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Server runs at http://localhost:5000

### Frontend
Open `frontend/index.html` in your browser directly, or serve with:
```bash
cd frontend
python -m http.server 8080
```
Then visit http://localhost:8080

## Training Your Own Model

1. Download dataset from Kaggle: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
2. Organize into folders by disease class
3. Run training:

```python
from model import train_model
train_model("path/to/your/dataset", epochs=20)
```

## Disease Classes
- Healthy
- Leaf Rust
- Leaf Spot
- Powdery Mildew
- Bacterial Blight

## Research References
See `research_resources.md` for all IEEE/journal citations.
