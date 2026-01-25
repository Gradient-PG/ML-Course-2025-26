from fastapi import FastAPI, File, UploadFile
from typing import Annotated
from garmet_classifier import GarmentClassifier
import cv2
import torch
import numpy as np

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


def preprocess(image):
    # min-max scaling
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    return GarmentClassifier.classes[class_id], 200


if __name__ == "__main__":
    model = GarmentClassifier()
    model.load_state_dict(torch.load("model_final"))
