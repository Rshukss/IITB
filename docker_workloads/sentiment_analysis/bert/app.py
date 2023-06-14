from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and tokenizer onto the device
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Define a function to perform sentiment analysis
def predict_sentiment(text):
    # Tokenize the input text and truncate to max length
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    # Move the inputs to the device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    # Run the inputs through the model
    outputs = model(**inputs)
    # Get the predicted label (0=negative, 1=neutral, 2=positive)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    # Return the predicted label and the confidence scores
    return predicted_label, outputs.logits.softmax(dim=1).tolist()[0]

# Define a route for the REST API
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_api():
    # Get the text input from the request
    text = request.json['text']
    # Perform sentiment analysis using the model
    label, scores = predict_sentiment(text)
    # Create a response object with the predicted label and scores
    response = {
        'sentiment_label': label,
        'sentiment_scores': scores
    }
    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5127', debug=True)

