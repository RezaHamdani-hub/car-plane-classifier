from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Car vs Plane Classifier")

# Static files (frontend UI) 
import os
STATIC_DIR = "app/static" if os.path.exists("app/static") else "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load model once at startup 
MODEL_PATH = "model/classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (64, 64)

# Must match train.py → {'airplanes': 0, 'cars': 1}
CLASS_NAMES = {0: "airplane", 1: "car"}

@app.get("/")
def root():
    html = STATIC_DIR + "/index.html"
    return FileResponse(html)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read & preprocess image 
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3)

    # Predict 
    prediction = model.predict(img_array)[0][0]     # float 0..1
    class_id   = int(prediction > 0.5)
    label      = CLASS_NAMES[class_id]
    confidence = float(prediction) if class_id == 1 else float(1 - prediction)

    return JSONResponse({
        "prediction": label,
        "confidence": f"{round(confidence * 100, 2)}%",
        "raw_score":  round(float(prediction), 4)
    })