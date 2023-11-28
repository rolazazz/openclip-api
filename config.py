from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(), verbose=True) 

MODEL_NAME = os.environ.get('MODEL_NAME')
MODEL_PRETRAINED = os.environ.get('MODEL_PRETRAINED')
MODEL_ID = os.environ.get('MODEL_ID')
CACHE_DIR = os.environ.get('CACHE_DIR')
