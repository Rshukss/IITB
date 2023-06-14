# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import load_model
# from tensorflow.keras.datasets import imdb
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     print('GPU is available!')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# else:
#     print('GPU is not available')
    
# # Load the pre-trained model
# model = load_model('fastText_imdb')
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Define a function to encode text input into a numerical sequence
# def encode_text(text):
#     tokenizer = Tokenizer(num_words=10000)
#     tokenizer.fit_on_texts(text)
#     sequences = tokenizer.texts_to_sequences(text)
#     return pad_sequences(sequences, maxlen=100)

# # Define a function to predict the sentiment of a text input
# def predict_sentiment(text, threshold=0.5):
#     encoded_text = encode_text([text])
#     pred = model.predict(np.array(encoded_text))[0][0]
#     return 'Positive' if pred >= threshold else 'Negative'


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the text input from the JSON body of the request
#     text = request.json['text']

#     # Predict the sentiment
#     sentiment = predict_sentiment(text)

#     # Return the predicted sentiment as a JSON response
#     response = {
#         'sentiment': sentiment
#     }
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=4000)


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from flask import Flask, request, jsonify

app = Flask(__name__)

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print('GPU is available!')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print('GPU is not available')
    
# Load the pre-trained model
model = load_model('fastText_imdb')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a function to encode text input into a numerical sequence
def encode_text(text):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    return pad_sequences(sequences, maxlen=100)

# Define a function to predict the sentiment of a text input
def predict_sentiment(text, threshold=0.5):
    encoded_text = encode_text([text])
    pred = model.predict(np.array(encoded_text))[0][0]
    return 'Positive' if pred >= threshold else 'Negative'


@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the JSON body of the request
    text = request.json['text']

    # Predict the sentiment
    sentiment = predict_sentiment(text)

    # Return the predicted sentiment as a JSON response
    response = {
        'sentiment': sentiment
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5125)



