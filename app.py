import json
import os

import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import IMG_SIZE, build_model

DATA_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(DATA_DIR, "model.keras")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.json")
EPOCHS = 20

state = {"model": None, "class_names": None}


def load_if_exists():
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        state["model"] = keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH) as f:
            state["class_names"] = json.load(f)


def train():
    dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=4,
        shuffle=True,
    )
    class_names = dataset.class_names

    model = build_model(num_classes=len(class_names))
    history = model.fit(dataset, epochs=EPOCHS, verbose=0)

    model.save(MODEL_PATH)
    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names, f)

    state["model"] = model
    state["class_names"] = class_names

    final_acc = history.history["accuracy"][-1]
    return f"Training done. Classes: {class_names}. Final accuracy: {final_acc:.2f}"


def predict(image):
    if state["model"] is None:
        return {"Train the model first": 1.0}

    img = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    img = tf.expand_dims(img, axis=0)

    preds = state["model"].predict(img, verbose=0)[0]
    return {name: float(prob) for name, prob in zip(state["class_names"], preds)}


load_if_exists()

with gr.Blocks() as demo:
    gr.Markdown("# Custom Keras Image Classifier")

    with gr.Tab("Train"):
        gr.Markdown(
            "Put ~10 images of each object into the `object1/` and `object2/` "
            "folders next to this script, then click Train."
        )
        train_btn = gr.Button("Train")
        train_output = gr.Textbox(label="Status")
        train_btn.click(fn=train, outputs=train_output)

    with gr.Tab("Predict"):
        image_input = gr.Image(type="numpy", label="Image")
        predict_btn = gr.Button("Predict")
        predict_output = gr.Label(label="Prediction")
        predict_btn.click(fn=predict, inputs=image_input, outputs=predict_output)

if __name__ == "__main__":
    demo.launch()
