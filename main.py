from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from keras.models import load_model
import numpy as np
import pickle
import re

"""
1. What constant should we make ?
    A. Model Path
    B. Tokenizer Path
    C. Max Sequence Length
    D. Emotion Labels
    E. Emotion emojis 
"""

model_path = "Artifacts/BiGRU_Model.keras"
tokenizer_path = "Artifacts/tokenizer.pkl"
max_sequence_length = 50
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']   
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

"""
2. Preprocessing the text
    Convert the text to lowercase, remove special characters, and extra spaces.
"""

def preprocess_text(text: str)->str:
    text = text.lower()
    text = re.sub(r"'","",text)
    text = re.sub(r"[^a-z0-9\s]"," ", text)
    text = re.sub(r"\s+", " ",text).strip()
    return text

"""
3. Request and Response Schemas
    Take input sent by the user i.e., input schema
    Return prediction response 
    Server performance tracking 
"""

class TextInput(BaseModel):
    text : str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"}
        )

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence : float
    all_probabilites: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


"""
4. Model Loading and LifeSpan Management
    Load the model and tokenizer when the server starts and clear them when the server stops.
"""
dl_model = {} #{1. BiGRU, 2. Tokenizer}-> True , {} -> False

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading the model and tokenizer...')
    dl_model["BiGRU"] = load_model(model_path)                   
    with open(tokenizer_path, 'rb') as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print('Model are loaded successfully...')   

    # Pause, model is laoded and server is running and at this point. Model is waiting 
    yield 

    # Clear the model and tokenizer when the server stops  
    dl_model.clear()

"""
5. Mount the static files to the FastAPI app
    Enable CORS (Cross-Origin Resource Sharing) to allow requests from different origins.
"""
app =  FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")

"""
6. API Endpoints.
    Server UI at Homepage
    Health Check Endpoint
    Predict Emotion Endpoint
"""

# Server UI at Homepage
@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse("static/index.html")


# Health Check Endpoint
@app.get("/health", response_model = HealthResponse)
def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))


# Predict Emotion Endpoint
@app.post("/predict", response_model=PredictionResponse)

def predict_emotion(text_input: TextInput):
    """
    # Cleans the input sentence 
    # Converts the words into the numeric using tokenizer
    # Pad sequence to make the length of the sentence equal to max_sequence_length 
    # Run prediction using the BiGRU
    # Return the top emotion
    """
    BiGRU_model     = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    #1. Clean the input sentence
    cleaned_text = preprocess_text(text_input.text)

    #2. Convert the words into numeric using tokenizer 
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])

    #3. Pad sequence to make the length of the sentence equal to max_sequence_length
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    #4. Run prediction using the BiGRU
    probabilites     = BiGRU_model.predict(padded_sequence)[0]

    #5. Return the top emotion
    top_emotion_index = int(np.argmax(probabilites))
    all_probabilites =  {
        label: float(prob) for prob, label in zip(probabilites, emotion_labels)
          
    }

    return PredictionResponse(
        text = text_input.text,
        predicted_emotion = emotion_labels[top_emotion_index],
        confidence = float(probabilites[top_emotion_index]), 
        all_probabilites = all_probabilites
    )