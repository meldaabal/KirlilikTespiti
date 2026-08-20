import os
import numpy as np
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


app = FastAPI(title="Kirlilik Tespiti API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "../models/mobilenetv2_transfer_learning.keras"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

model = tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    return image_array


@app.get("/")
def home():
    return {
        "message": "Kirlilik Tespiti API çalışıyor."
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    processed_image = prepare_image(image_bytes)

    predictions = model.predict(processed_image)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    predicted_class = CLASS_NAMES[predicted_index]

    return {
        "class": predicted_class,
        "confidence": round(confidence * 100, 2)
    }