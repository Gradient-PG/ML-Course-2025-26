from flask import Flask, request, send_from_directory
from garmet_classifier import GarmentClassifier
import torch
import cv2

app = Flask(__name__)


def preprocess(image):
    # min-max scaling
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = torch.from_numpy(image).reshape([1, 28, 28]).to(torch.float32)
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 2 - 1
    return image


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    file.save(f"tmp/{file.filename}")
    image = cv2.imread(f"tmp/{file.filename}")
    image = preprocess(image)

    output: torch.Tensor = model.forward(image)

    class_id = int(output.argmax())

    return GarmentClassifier.classes[class_id], 200


if __name__ == "__main__":
    model = GarmentClassifier()
    model.load_state_dict(torch.load("model_final"))
    app.run(debug=True)
