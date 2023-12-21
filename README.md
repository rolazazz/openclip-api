---
title: OpenCLIP API
emoji: 🌍
colorFrom: red
colorTo: red
sdk: docker
pinned: false
license: mit
---


# OpenCLIP-API
[**CLIP**](https://openai.com/research/clip) (Contrastive Language-Image Pre-training) is a method created by OpenAI for training models capable of aligning image and text representations. Images and text are drastically different modalities, but CLIP manages to map both to a shared space, allowing for all kinds of neat tricks.

[**OpenCLIP**](https://github.com/mlfoundations/open_clip) is an open-source replication effort that has successfully created a CLIP implementation and released a number of trained models for anyone to use.
After a couple of tests OpenCLIP gave a better accuracy, probably due to a larger dataset used for pretraing ([LAION-2B](https://laion.ai/projects/)).

[**FastAPI**](https://fastapi.tiangolo.com/) is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.

## Embeddings

### What are embeddings?
Embeddings are representations of values or objects like text, images, and audio that are designed to be consumed by machine learning models and semantic search algorithms.
Essentially, embeddings enable machine learning models to find similar objects. Given a photo or a document, a machine learning model that uses embeddings could find a similar photo or document. Since embeddings make it possible for computers to understand the relationships between words and other objects, they are foundational for many ML tasks like **Image Similarity Search**.

### Get the embbedings
The API expose one singular endpoint **/embeddings** and accept both text or image. the property "***image***" must be a base64 econded string of a image stream.
```http
POST /embeddings HTTP/1.1
Host: localhost:7860
Content-Type: application/json
Content-Length: 18309

{
	"text": "red sofa",
    "image": "/9j/4AAQSkZJRgABAQAAAQABAAD/[...]"
}
```
or
```bash
$ curl --location --request POST 'http://localhost:7860/embeddings' \
--header 'Content-Type: application/json' \
--data-raw '{
	"text": "red sofa",
    "image": "/9j/4AAQSkZJRgABAQAAAQABAAD/[...]"
	}'
```

## Requirements
### Python environment
**Python v3.9+**

Make sure, you can access and use Python.

## How to

### Setup Python Virtual Evironment (.venv)
We must set up a Python environment to use scripts for the api.
A virtual environment, a self-contained directory tree that contains a Python installation for a particular version of Python, plus a number of additional packages. 
```bash
$ git clone https://github.com/rolazazz/openclip-api
$ cd openclip-api
$ python -m venv .venv
$ ./.venv/bin/activate
$ pip install -r requirements.txt
```

## Run the API locally
Make sure that Python environment is set and all requirements are installed as described above and that you are in the main project folder.
```bash
# In the main directory 
$ python main.py
```



## How to run OpenCLIP/FastAPI in Docker 
To run the application in a Docker container, we need to build it and then run the Docker image with the OpenCLIP/FastAPI application.
```bash
$ # just make sure you are in the main project directory 
$ cd openclip-fastapi
````

### Build the image
In order to be able to run the application in the Docker environment, we need to build the image locally. Because this 
is a Python application with dependencies, the build of the image might take longer. All the requirements are installed.   
```bash
$ docker build . --tag openclip-api:0.0.1 --tag openclip-api:latest
```
Once, the build is complete, we can verify if the image is available.
```bash
$ docker images | grep openclip-api
```

### Run the image
To run the application, we need to run the Docker image. 
```bash
$ docker run -d -p 7860:7860 --name openclip-api [--volume openclip-cache:/home/user/.cache] openclip-api:latest
```

### Important notes

When the application starts up, it needs to download a model from huggingface.co, which will take some time.
The total time the application start takes depends on your network connection.
It's hightly recommended to mount a persisted volume on user's ***.cache*** folder 

You might see in the terminal similar output.
```bash
loading model OpenCLIP-ViT-B-32-laion2B...
open_clip_pytorch_model.bin:   0%|          | 0.00/605M [00:00<?, ?B/s]
open_clip_pytorch_model.bin:  10%|█         | 62.9M/605M [00:00<00:08, 62.6MB/s]
open_clip_pytorch_model.bin:  49%|████▊     | 294M/605M [00:05<00:05, 53.6MB/s]
open_clip_pytorch_model.bin:  74%|███████▍  | 451M/605M [00:08<00:02, 55.4MB/s]
open_clip_pytorch_model.bin: 100%|██████████| 605M/605M [00:10<00:00, 56.4MB/s]
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
INFO:     10.16.34.18:45435 - "GET / HTTP/1.1" 200 OK
```
Once the application is up and running, you will something like this calling the endpoint **/embeddings**
```bash
2023-11-28 14:57:50 INFO:     172.17.0.1:43486 - "POST /embeddings HTTP/1.1" 200 OK
2023-11-28 14:58:01 INFO:     172.17.0.1:39600 - "POST /embeddings HTTP/1.1" 200 OK
```

### Environment variables
You can change the OpenCLIP model version setting some environment variables:
- MODEL_NAME = 'ViT-B-32' (default)
- MODEL_PRETRAINED = 'laion2b_s34b_b79k' (default)
- MODEL_ID = 'OpenCLIP-ViT-B-32-laion2B' (default)

With docker you must run the container adding the option **-e** or **--env**:
```bash
$ docker run -d -p 7860:7860 --name openclip-api -e MODEL_NAME='ViT-L-14' -e MODEL_PRETRAINED='laion2b_s32b_b82k' -e MODEL_ID='OpenCLIP-ViT-L-14-laion2B' --volume openclip-cache:/home/user/.cache openclip-api:latest
```

All available OpenCLIP models are listed on [laion's collection](https://huggingface.co/collections/laion/openclip-laion-2b-64fcade42d20ced4e9389b30) @huggingface.