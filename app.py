from config import MODEL_NAME, MODEL_PRETRAINED, MODEL_ID, CACHE_DIR
from PIL import Image
import torch, base64, io, time
import open_clip


class InferlessPythonModel:

    # Implement the Load function here for the model
	def initialize(self):
		print(f"loading model {MODEL_ID}...")
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=MODEL_PRETRAINED, cache_dir=CACHE_DIR, device=self.device)
		self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)


	# Function to perform inference 
	def infer(self, inputs):
		# inputs is a dictonary where the keys are input names and values are actual input data
		# e.g. in the below code the input name is "prompt"
		# prompt = inputs["prompt"]	

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
			"model": MODEL_ID,
			"device": self.device.type,
			# "inputs" : inputs,
			"embeddings": text_features + image_features,
			"duration": time.perf_counter() - start_time
			}

	# perform any cleanup activity here
	def finalize(self, args):
		self.device, self.model, self.preprocess, self.tokenizer = None
