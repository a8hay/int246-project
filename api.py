from flask import Flask, jsonify, request

import io
import json

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

def preprocess_image(image_bytes, target_size):
	image = Image.open(io.BytesIO(image_bytes))
	if image.mode != "RGB":
		image = image.convert("RGB")
	transform = transforms.Compose([
			transforms.Resize(target_size),
			transforms.ToTensor()
		])
	image = transform(image).unsqueeze(dim=0)
	return image

def get_prediction(image_bytes):
	processed_image  = preprocess_image(image_bytes, target_size=(128, 128))
	with torch.no_grad():
		output = model(processed_image)
		pred = torch.argmax(output, dim=1)
		res =  "dog" if pred.item() else "cat"
	return res
ALLOWED_EXTENSIONS = {"png", "jpeg", "jpg"}
def allowed_file(filename):
	# abc.png, abc.jpeg, abc.jpg
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello():
    return "hiya"

@app.route('/predict', methods=["POST"])
def predict():
	if request.method == "POST":
		file = request.files.get("file")
		if file is None or file.filename == "":
			return jsonify({"error":"no file"})
		if not allowed_file(file.filename):
			return jsonify({"error":"formated not supported, only jpg, png, jpeg"})
		try:
			img_bytes = file.read()
			pred = get_prediction(img_bytes)
			return jsonify({"prediction":pred})
		except:
			return jsonify({"error":"error during prediction process"})
	return jsonify({"bad request":"request is not of post type"})

if __name__ == "__main__":
	print("loading pytorch model")
	get_model()
	app.run(debug=False)