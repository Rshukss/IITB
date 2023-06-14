from flask import Flask, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

device = torch.device('cuda')
print("Using device:", device)

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the text from the JSON body of the request
    text = request.json['text']

    # Generate the summary
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)

    # Return the summary as a JSON response
    response = {
        'summary': summary[0]['summary_text']
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5555)
