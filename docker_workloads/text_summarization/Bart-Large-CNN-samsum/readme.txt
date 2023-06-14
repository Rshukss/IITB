Variant-
    * Text Summarization
    * Bart Large CNN - samsum
    * Pytorch

List of commands-
    * Build - sudo docker build -t new_img_summarization -f Dockerfile_new_summarization .
    * Run - sudo docker run -d --gpus all -p 5555:5555 --runtime=nvidia new_img_summarization
    * Test - 
        Input - curl --location --request POST 'http://localhost:5555/summarize' \
                --header 'Content-Type: application/json' \
                --data-raw '{
                    "text": "Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker? Philipp: Sure you can use the new Hugging Face Deep Learning Container. Jeff: ok. Jeff: and how can I get started? Jeff: where can I find documentation? Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face"
                }'

        Ouput - {
                "summary": "Jeff wants to train a Transformers model on Amazon SageMaker. Philipp says he can use the new Hugging Face Deep Learning Container. Jeff can find the documentation here."
                }

List of items in dockerfile that can be changed-
    * Base image should be nvidia/cuda:12.1.0-runtime-ubuntu20.04, on 18.04 there were many version errors due to python3.6 and pip9
    * timm package needs to be installed(pipreqs cannot capture this package from app.py)
    * pipreqs --mode no-pin does not list the versions along with the package
                    (but on ubuntu20.04 should work on default mode - NOT TESTED)