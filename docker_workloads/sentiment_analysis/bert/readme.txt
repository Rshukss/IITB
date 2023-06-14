Variant-
    * Sentiment analysis 
    * BERT
    * Pytorch

List of commands-
    * Build - sudo docker build -t gpu-sentiment-analysis-bert .
    * Run - sudo docker run -d --gpus all -p 5127:5127 --runtime=nvidia gpu-object-detection
    * Test - 
        Input - curl -X POST -H "Content-Type: application/json" -d '{"text": "I love this product, it works great!"}' http://localhost:5127/predict_sentiment

        Ouput - {
  "sentiment_label": 2,
  "sentiment_scores": [
    0.0019167434656992555,
    0.005206411704421043,
    0.9928768277168274
  ]
}


List of items in dockerfile that can be changed-
    * Base image should be nvidia/cuda:12.1.0-runtime-ubuntu20.04, on 18.04 there were many version errors due to python3.6 and pip9
    * timm package needs to be installed(pipreqs cannot capture this package from app.py)
    * pipreqs --mode no-pin does not list the versions along with the package
                    (but on ubuntu20.04 should work on default mode - NOT TESTED)
