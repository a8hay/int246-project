from flask import Flask, jsonify

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image

app = Flask(__name__)

def get_model():
	global model
	model = torchvision.models.densenet121(pretrained=True)
	num_ftrs = model.classifier.in_features
	model.classifier = nn.Sequential(
			nn.Linear(num_ftrs, 500),
			nn.Linear(500, 2)
		)
	model.load_state_dict(torch.load("ckpt_densenet121_catdog.pth", map_location=torch.device("cpu")))
	model.to("cpu")
	model.eval()
	print("model loaded")

@app.route('/')
def hello():
    return "hiya"

@app.route('/predict', methods=["POST"])
def predict():
    return jsonify({"cat":0, "dog":1})

if __name__ == "__main__":
    print("loading pytorch modal")
    get_model()
    app.run(debug=False)