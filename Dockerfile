# containerized api to get embeddings from OpenCLIP
# when the app starts, the model binaries will be be downloaded from huggingface.co
# it's hightly recommended to mount a persisted volume on
# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# the part below makes the container compatible with huggingface.com
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "7860"]