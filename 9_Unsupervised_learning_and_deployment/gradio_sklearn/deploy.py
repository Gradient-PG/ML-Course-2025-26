import gradio as gr
import pandas as pd
import pickle

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

with open("data/kmeans.pickle", "rb") as rfile:
    kmeans = pickle.load(rfile)

def predict_species(sepal_len, sepal_width, petal_len, petal_width):
    cluster_id = kmeans.predict([[sepal_len, sepal_width, petal_len, petal_width]])[0]
    return f" Specified flower belongs to {cluster_id}"

demo = gr.Interface(
    predict_species,
    [
        gr.Slider(0, 10, label = "sepal_len"),
        gr.Slider(0, 10, label = "sepal_width"),
        gr.Slider(0, 10, label = "petal_len"),
        gr.Slider(0, 10, label = "petal_width"),
    ],
    outputs="text",
)

demo.launch(share=True)
