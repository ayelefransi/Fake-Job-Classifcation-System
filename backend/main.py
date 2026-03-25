from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import re
from contextlib import asynccontextmanager

# ──────────────────────────────────────────────
# Load models at startup
# ──────────────────────────────────────────────
ensemble_model = None
bert_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ensemble_model, bert_model
    print("Loading models…")
    model_path = os.path.join(os.path.dirname(__file__), "..", "Models", "fake_job_ensemble.joblib")
    ensemble_model = joblib.load(model_path)

    from sentence_transformers import SentenceTransformer
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Models loaded successfully.")
    yield
    print("Shutting down.")

app = FastAPI(title="Fake Job Detector API", version="1.0.0", lifespan=lifespan)

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────
class JobInput(BaseModel):
    title: str = ""
    description: str = ""
    requirements: str = ""

class PredictionResult(BaseModel):
    probability: float
    label: str          # "LEGITIMATE" | "SUSPICIOUS" | "FRAUDULENT"
    confidence: str     # e.g. "87.3%"
    features: dict

# ──────────────────────────────────────────────
# Feature engineering (mirrors app.py exactly)
# ──────────────────────────────────────────────
SCAM_KEYWORDS = [
    "urgent", "easy money", "no experience", "wire transfer",
    "western union", "cash"
]

def build_features(title: str, description: str, requirements: str) -> pd.DataFrame:
    full_text = f"{title} {description} {requirements}"
    text_lower = full_text.lower()

    # Structural features
    text_length = len(full_text)
    has_requirements = 1 if len(requirements.strip()) > 5 else 0
    scam_keyword_count = sum(1 for kw in SCAM_KEYWORDS if kw in text_lower)

    features: dict = {
        "text_length": [text_length],
        "has_requirements": [has_requirements],
        "scam_keyword_count": [scam_keyword_count],
    }

    # BERT embeddings (384 dims)
    embeddings = bert_model.encode([full_text])  # shape (1, 384)
    for i in range(embeddings.shape[1]):
        features[f"bert_{i}"] = [float(embeddings[0][i])]

    df = pd.DataFrame(features)
    # Align columns to model's expected order
    df = df[ensemble_model.feature_names_in_]
    return df, {
        "text_length": text_length,
        "has_requirements": bool(has_requirements),
        "scam_keyword_count": scam_keyword_count,
    }

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Fake Job Detector API is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": ensemble_model is not None,
        "bert_loaded": bert_model is not None,
    }

@app.post("/predict", response_model=PredictionResult)
def predict(job: JobInput):
    if ensemble_model is None or bert_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    if not job.title.strip() and not job.description.strip():
        raise HTTPException(status_code=400, detail="Provide at least a title or description")

    input_df, insight = build_features(job.title, job.description, job.requirements)
    prob = float(ensemble_model.predict_proba(input_df)[0][1])

    if prob > 0.65:
        label = "FRAUDULENT"
    elif prob > 0.40:
        label = "SUSPICIOUS"
    else:
        label = "LEGITIMATE"

    return PredictionResult(
        probability=round(prob, 4),
        label=label,
        confidence=f"{prob:.1%}",
        features=insight,
    )
