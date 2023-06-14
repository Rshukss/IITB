Variant-
    * Image recognition 
    * Resnet50
    * Pytorch

List of commands-
    * Build - sudo docker build -t gpu-image-recognition-resnet .
    * Run - sudo docker run -d --gpus all -p 5124:5124 --runtime=nvidia --ipc=host gpu-image-recognition-resnet
    * Test - 
        Input - curl -X POST -H "Content-Type: application/json" -d '{"url": "http://images.cocodataset.org/val2017/000000039769.jpg"}' http://localhost:5124/predict
        Ouput - {
                    "class": "tabby",
                    "score": 8.045576095581055
                }   


List of items in dockerfile that can be changed-
    * Base image should be nvidia/cuda:12.1.0-runtime-ubuntu20.04, on 18.04 there were many version errors due to python3.6 and pip9
    * timm package needs to be installed(pipreqs cannot capture this package from app.py)
    * pipreqs --mode no-pin does not list the versions along with the package
                    (but on ubuntu20.04 should work on default mode - NOT TESTED)
