from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
import onnxruntime as ort
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

try:
    onnx_session = ort.InferenceSession(
        "app/model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
except Exception as e:
    print(f"CUDA not available or failed to initialize: {e}. Falling back to CPU.")
    onnx_session = ort.InferenceSession(
        "app/model.onnx", providers=["CPUExecutionProvider"]
    )

garmet_classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


def preprocess(image):
    # min-max scaling
    image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 2 - 1
    image = image.reshape(1, 28, 28)
    return image


@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    image = preprocess(image)

    ort_inputs = {onnx_session.get_inputs()[0].name: image}
    ort_outs = onnx_session.run(None, ort_inputs)
    output = ort_outs[0]

    class_id = int(np.argmax(output))

    print(f"Predicted class id: {class_id}")

    return garmet_classes[class_id], 200
