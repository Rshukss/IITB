from flask import Flask, request, jsonify
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torch
import requests
from PIL import Image

app = Flask(__name__)

# Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

device = torch.device('cuda')
print("Using device:",device)

# Ensure that the model runs on GPU if available
model.to(device)

# Initialize the inference transforms
preprocess = weights.transforms()

@app.route('/predict', methods=['POST'])
def predict():
    # Get the URL of the image from the JSON body of the request
    url = request.json['url']

    # Download the image and convert it to a PyTorch tensor
    response = requests.get(url)
    with open("image.jpg", "wb") as f:
        f.write(response.content)
    img = Image.open('image.jpg').convert('RGB')
    img = preprocess(img).unsqueeze(0)

    # Move the input tensor to the same device as the model
    img = img.to(device)

    # Make a prediction using the ResNet50 model
    with torch.no_grad():
        output = model(img)

    # Get the predicted class and score
    class_id = output[0].argmax().item()
    score = output[0][class_id].item()
    category_name = weights.meta["categories"][class_id]

    # Return the predicted class and score as a JSON response
    response = {
        'class': category_name,
        'score': score
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5124)
