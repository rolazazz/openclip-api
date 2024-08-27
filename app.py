from config import CLIP_MODEL_NAME, CLIP_MODEL_PRETRAINED, CLIP_MODEL_ID, E5_MODEL_NAME, CACHE_DIR
from PIL import Image
import torch, base64, io, time
import open_clip
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class ClipInferlessPythonModel:

    # Implement the Load function here for the model
	def initialize(self):
		print(f"loading model {CLIP_MODEL_ID}...")
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model, _, self.preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_MODEL_PRETRAINED, cache_dir=CACHE_DIR, device=self.device)
		self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)


	# Function to perform inference 
	def infer(self, inputs):
		# inputs is a dictonary where the keys are input names and values are actual input data
		# e.g. in the below code the input name is "text"
		# prompt = inputs["text"]	

		if (inputs.get("text") == None and inputs.get("image") == None):
			raise ValueError("the parameter 'Inputs' must contain 'text' or 'image' value") 

		start_time = time.perf_counter()
		text_features = image_features = []

		# get text embbeddings
		if (inputs.get("text") != None):
			text = self.tokenizer(inputs.get("text")).to(self.device)
			text_features = self.model.encode_text(text).tolist()

		# get image embeddings
		if (inputs.get("image") != None):
			# convert it into bytes  
			im_bytes = base64.b64decode(inputs.get("image").encode('utf-8'))
			# convert bytes data to PIL Image object
			im_obj = Image.open(io.BytesIO(im_bytes))

			clip_image = self.preprocess(im_obj).unsqueeze(0).to(self.device)
			image_features = self.model.encode_image(clip_image).tolist()

		return {	
			"model": CLIP_MODEL_ID,
			"device": self.device.type,
			# "inputs" : inputs,
			"embeddings": text_features + image_features,
			"duration": time.perf_counter() - start_time
			}


	# perform any cleanup activity here
	def finalize(self, args):
		self.device, self.model, self.preprocess, self.tokenizer = None



class E5InferlessPythonModel:

	def initialize(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
		# self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to("cuda")
		self.st = SentenceTransformer('intfloat/multilingual-e5-large', device=self.device, cache_folder=CACHE_DIR)


	# def mean_pooling(self, model_output, attention_mask):
	# 	token_embeddings = model_output[0]
	# 	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	# 	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


	def infer(self, inputs):
		# inputs is a dictonary where the keys are input names and values are actual input data
		# e.g. in the below code the input name is "text"
		# prompt = inputs["text"]	

		if (inputs.get("text") == None ):
			raise ValueError("the parameter 'Inputs' must contain 'text'") 

		start_time = time.perf_counter()

		# get text embbeddings
		if (inputs.get("text") != None):
			# encoded_text = self.tokenizer(inputs.get("text"), padding=True, truncation=True, return_tensors='pt').to(self.device)
			# with torch.no_grad():
			# 	model_output = self.model(**encoded_text)
			# features = self.mean_pooling(model_output, encoded_text['attention_mask'])
			features = self.st.encode(inputs.get("text"), convert_to_tensor=True, device=self.device).tolist()

		return {	
			"model": E5_MODEL_NAME,
			"device": self.device.type,
			# "inputs" : inputs,
			"embeddings": [features],
			"duration": time.perf_counter() - start_time
			}


	def finalize(self, args):
		self.model = None
		self.tokenizer = None