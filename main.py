import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import  JSONResponse
from pydantic import BaseModel, Field
from app import InferlessPythonModel




app = InferlessPythonModel()
app.initialize()

api = FastAPI()


class Inputs(BaseModel):
    """
    Class for input endpoint /embeddings.

    Args:
        text (str): string text to extract the embeddings from.
        image (str): image base64 encoded.
    """
    text: str | None = None
    image: str | None = None
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "red sofa",
                    "image": "base64 image.."
                }
            ]
        }
    }


@api.get("/", tags=["root"])
def read_root():
	return {"message": "archiproducts.ai!"}


@api.post("/embeddings")
def get_embeddings(inputs:Inputs, request: Request):
	"""
    Get embeddings vector form input text or image.
    Args:
        input_data (Inputs): Oggetto contain text or image.
    Returns:
        dict: result containing the embeddings.
    """
	return app.infer(inputs.__dict__)
	

if __name__ == "__main__":
	uvicorn.run(api, host='127.0.0.1', port=8000)
	# uvicorn.run(app="main:api", host='127.0.0.1', port=8000)
