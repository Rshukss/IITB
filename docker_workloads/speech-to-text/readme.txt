Variant-
    * Speech-to-Text
    * Silerio
    * Tensorflow

List of commands-
    * Build - sudo docker build -t gpu-stt -f Dockerfile_tf .
    * Run - sudo docker run -d --gpus all -p 3333:3333 --runtime=nvidia --ipc=host gpu-stt
    * Test - 
        Input - curl -X POST -H "Content-Type: application/json" -d '{"url": "Sample URL link (of a .wav file)"}' http://localhost:3333/transcribe

        Ouput - Transcribed text


List of items in dockerfile that can be changed-
    * timm package needs to be installed(pipreqs cannot capture this package from app.py)
    * pipreqs --mode no-pin does not list the versions along with the package
                    (but on ubuntu20.04 should work on default mode - NOT TESTED)
