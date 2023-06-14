Variant-
    * Image recognition 
    * vgg16
    * Tensorflow

List of commands-
    * Build - sudo docker build -t gpu-image-recognition-vgg-tensorflow .
    * Run - sudo docker run -d --gpus all -p 3333:3333 --runtime=nvidia gpu-image-recognition-vgg-tensorflow
    * Test - 
        Input - curl -X POST -H "Content-Type: application/json" -d '{"image_url": "http://images.cocodataset.org/val2017/000000039769.jpg"}' http://localhost:3333/predict
        Ouput - Banana

List of items in dockerfile that can be changed-
    * Base image should be tensorflow/tensorflow-gpu, no need for cuda base images.
