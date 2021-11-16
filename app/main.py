"""
Here I used fast API which is a simple and faster API platform.
"""

from fastapi import FastAPI, Depends
from app.config import APP_NAME, APP_VERSION, IS_DEBUG
import json, os
from pydantic import BaseModel
import numpy as np
from fastapi.responses import JSONResponse
import tensorflow_hub as hub

embed=hub.load('./app/model/thub')


class Input_for_sentence_similarity(BaseModel):
    sentence1: str
    sentence2: str


def get_app() -> FastAPI:
    fast_app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        debug=IS_DEBUG,
        redoc_url=None,
        docs_url=None,
    )

    return fast_app


app = get_app()


@app.post("/sentence_similarity")
def sentence_similarity(
    incoming_data: Input_for_sentence_similarity,
    status_code=200,
):
    data = incoming_data.dict()
    sentence1 = data["sentence1"]
    sentence2=data["sentence2"]
    sentences=[sentence1,sentence2]
    embedded_sentences=embed(sentences)
    similarity=np.inner(embedded_sentences[0],embedded_sentences[1])
    return JSONResponse(float(similarity))