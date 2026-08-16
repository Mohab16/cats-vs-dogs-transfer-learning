from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from pydantic import BaseModel

app = FastAPI()

model = tf.keras.models.load_model(
    "models/final_training.keras"
)


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probability: float

@app.get("/")
def home():
    return {"message": "Cats vs Dogs API is running!"}

@app.post("/predict",response_model=PredictionResponse)
async def predict (file: UploadFile = File()):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image=image.resize((160,160))
    image_array = np.array(image)
    image_array = np.expand_dims(image, axis=0)
    
    probability = model.predict(image_array, verbose = 0)[0][0]
    
    if probability >= 0.5:
        label = "Dog"
        confidence = probability
    else:
        label = "Cat"
        confidence = 1 - probability
        
    return PredictionResponse(
        label=label,
        confidence= float(confidence),
        probability= float(probability),
    )

    