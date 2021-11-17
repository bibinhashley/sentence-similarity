# importing modules

from fastapi import FastAPI #fastapi module
from pydantic import BaseModel # used to validate the input from userimport numpy as np
import tensorflow_hub as hub
from fastapi.responses import JSONResponse # used for sending the output back to user.
import numpy as np

#loading model
model=hub.load("./app/model")

#defining how the input should look like
class Input_for_sentence_similarity(BaseModel):
     sentence1: str
     sentence2: str

# function to start fastapi APP
def get_app() -> FastAPI:
    fast_app = FastAPI(
        title="Sentence Similarity",
        version="0.0.1",
        debug=False,
        redoc_url=None,
        docs_url=None,) #if docs_url= True, you can try the api in localhost/docs
    return fast_app

app = get_app()


# function which we created before for sentance similarity checking.
@app.post("/sentence_similarity")
def sentence_similarity(
     incoming_data: Input_for_sentence_similarity,
     status_code=200,):
      
      # converting the user input to dictionary
      data = incoming_data.dict()
      
      #seperating both sentences to two variables
      sentence1 = data['sentence1']
      sentence2 = data['sentence2']
      
      # for converting to vectors, first add them to a list
      sentences = [sentence1,sentence2]
      
      # converting to vectors
      sentence_vectors = model(sentences)
      
      #finding similarity and sending as output
      similarity = np.inner(sentence_vectors[0],sentence_vectors[1])
      return JSONResponse(float(similarity))