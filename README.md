# Car vs Plane Classifier 🚀

A machine learning API that classifies images as **car** or **airplane**.

## Tech Stack
- TensorFlow/Keras (CNN model)
- FastAPI
- Docker
- Deployed on Render.com

## Run Locally
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

## API Usage

### GET /
Returns API status.

### POST /predict
Upload an image to classify.
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_image.jpg"
```

### Response
```json
{
  "prediction": "car",
  "confidence": "99.56%",
  "raw_score": 0.9956
}
```
