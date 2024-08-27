from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(), verbose=True) 

CLIP_MODEL_NAME = os.environ.get('CLIP_MODEL_NAME')
CLIP_MODEL_PRETRAINED = os.environ.get('CLIP_MODEL_PRETRAINED')
CLIP_MODEL_ID = os.environ.get('CLIP_MODEL_ID')

E5_MODEL_NAME = os.environ.get('E5_MODEL_NAME')

CACHE_DIR = os.environ.get('CACHE_DIR')
