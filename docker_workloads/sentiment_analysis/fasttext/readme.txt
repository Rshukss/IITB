Variant-
    * Sentiment analysis 
    * fasttext imdb
    * Tensorflow

List of commands-
    * Build - sudo docker build -f Dockerfile_tf -t gpu-sentiment-analysis-fasttext-tensorflow .
    * Run - sudo docker run -d --gpus all -p 5125:5125 --runtime=nvidia gpu-sentiment-analysis-fasttext-tensorflow
    * Test - 
        Input - curl -X POST -H "Content-Type: application/json" -d '{"text": "I really enjoyed the movie!"}' http://localhost:5125/predict
        Ouput - Positive

List of items in dockerfile that can be changed-
    * Base image should be tensorflow/tensorflow-gpu, no need for cuda base images.
    * The fastText_imdb folder contains the pre-trained model.
