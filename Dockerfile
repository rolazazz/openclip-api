# Containerized api to get embeddings from OpenCLIP
# When the application starts up, it needs to download a model from huggingface.co, which will take some time.
# The total time the application start takes depends on your network connection.
# It's hightly recommended to mount a persisted volume on user's '.cache' folder 
#
# BUILD the image: 
# docker build . --tag openclip-api:0.0.1 --tag openclip-api:latest --tag robertolazazzera/openclip-api:latest
# PUSH to registy
# docker push robertolazazzera/openclip-api:latest
# RUN the container:
# docker run -d -p 7860:7860 --name openclip-api openclip-api:latest
#
# The container is compatible with huggingface.co Spaces
# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

# FROM python:3.11-slim
FROM bitnami/pytorch:latest

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# required for bitnami
USER root 
# the part below makes the container compatible with huggingface.com
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN mkdir /home/user/.cache

EXPOSE 7860

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "7860"]