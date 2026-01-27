from fastapi import FastAPI, UploadFile
from .garmet_classifier import GarmentClassifier
import cv2
import torch
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = GarmentClassifier()
model.load_state_dict(torch.load("app/model_final"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (simplest for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


def preprocess(image):
    # min-max scaling
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = torch.from_numpy(image).reshape([1, 28, 28]).to(torch.float32)
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 2 - 1
    return image


@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    image = preprocess(image)

    output: torch.Tensor = model.forward(image)

    class_id = int(output.argmax())

    print(f"Predicted class id: {class_id}")

    return GarmentClassifier.classes[class_id], 200
