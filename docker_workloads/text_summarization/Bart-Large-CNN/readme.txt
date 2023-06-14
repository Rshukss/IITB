Variant-
    * Text Summarization
    * Bart Large CNN
    * Pytorch

List of commands-
    * Build - sudo docker build -t img_summarization -f Dockerfile_summarization .
    * Run - sudo docker run -d --gpus all -p 4444:4444 --runtime=nvidia --ipc=host img_summarization
    * Test - 
        Input - curl --location --request POST 'http://localhost:4444/summarize' \
                --header 'Content-Type: application/json' \
                --data-raw '{
                    "text": "Scientists have discovered a new species of dinosaur in China. The new species belongs to the theropod family, which includes other well-known dinosaurs like the T. rex and velociraptor. The researchers named the new species Haplocheirus sollers, which means \"simple-handed skillful hunter\". The dinosaur lived around 160 million years ago and had long, slender arms and a unique skull."
                }'
        Ouput - {
                "summary": "Scientists have discovered a new species of dinosaur in China. The new species belongs to the theropod family, which includes other well-known dinosaurs. The dinosaur lived around 160 million years ago and had long, slender arms and a unique skull."
                }

List of items in dockerfile that can be changed-
    * Base image should be nvidia/cuda:12.1.0-runtime-ubuntu20.04, on 18.04 there were many version errors due to python3.6 and pip9
    * timm package needs to be installed(pipreqs cannot capture this package from app.py)
    * pipreqs --mode no-pin does not list the versions along with the package
                    (but on ubuntu20.04 should work on default mode - NOT TESTED)