

# USE THIS CODE FINALLY

import torch
import torchaudio
import json
from flask import Flask, request, jsonify
import urllib.request
import requests

app = Flask(__name__)

device = torch.device('cuda')
print("Using device:",device)

# device = torch.device('cpu')
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

@app.route('/predict', methods=['POST'])
def transcribe():
    # Get the URL of the audio file from the JSON body of the request
    url = request.json['url']

    # Download the audio file and convert it to a PyTorch tensor
    # with urllib.request.urlopen(url) as url:
    #     response = url.read()
    # with open("audio.wav", "wb") as f:
    #     f.write(response)
    # URL = ""
    # 2. download the data behind the URL
    response = requests.get(url)

    # 3. Open the response into a new file called instagram.ico
    open("sample_audio.wav", "wb").write(response.content)
    audio, sample_rate = torchaudio.load("sample_audio.wav")
    audio = audio.mean(dim=0, keepdim=True)

    # Prepare the audio for input to the model
    input = prepare_model_input(audio, device=device)

    # Make a prediction using the Silero STT model
    with torch.no_grad():
        output = model(input)

    # Decode the predicted text
    text = decoder(output[0].cpu())

    # Return the predicted text as a JSON response
    response = {
        'text': text
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3333)

