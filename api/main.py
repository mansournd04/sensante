# api/main.py
# SenSante API - Assistant pre-diagnostic medical
# Lab 3 - Integration de Modeles IA - ESP / UCAD

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

from dotenv import load_dotenv

from groq import Groq

# ---------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------

class PatientInput(BaseModel):

    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)


class DiagnosticOutput(BaseModel):

    diagnostic: str
    probabilite: float
    confiance: str
    message: str

# ---------------------------------------------------
# Application FastAPI
# ---------------------------------------------------
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Charger .env

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# Initialiser Groq

groq_client = None

if api_key:

    groq_client = Groq(
        api_key=api_key
    )

    print("Client Groq initialise")

else:

    print("Cle API absente")
# ---------------------------------------------------
# Chargement du modele (une seule fois)
# ---------------------------------------------------

print("Chargement du modele...")

model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print(f"Modele charge : {list(model.classes_)}")

# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "message": "SenSante API is running"
    }

# ---------------------------------------------------
# Route POST /predict
# ---------------------------------------------------

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    # Encoder sexe
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]

    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}"
        )

    # Encoder region
    try:
        region_enc = le_region.transform([patient.region])[0]

    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # Features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # Prediction
    diagnostic = model.predict(features)[0]

    proba_max = float(
        model.predict_proba(features)[0].max()
    )

    # Niveau de confiance
    confiance = (
        "haute" if proba_max >= 0.7
        else "moyenne" if proba_max >= 0.4
        else "faible"
    )

    # Messages
    messages = {
        "palu": "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh": "Suspicion de typhoide. Consultation necessaire.",
        "sain": "Pas de pathologie detectee."
    }

    # Retour resultat
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(
            diagnostic,
            "Consultez un medecin."
        )
    )

class ExplainInput(BaseModel):

    diagnostic: str = Field(
        ...,
        description="Diagnostic predit par le modele"
    )

    probabilite: float = Field(
        ...,
        description="Probabilite du diagnostic"
    )

    age: int = Field(...)

    sexe: str = Field(...)

    temperature: float = Field(...)

    region: str = Field(...)


class ExplainOutput(BaseModel):

    explication: str = Field(
        ...,
        description="Explication en francais"
    )

    modele_llm: str = Field(
        default="llama-3.1-8b-instant",
        description="Modele LLM utilise"
    )
# ---------------------------------------------------
# Route GET /model-info
# ---------------------------------------------------

@app.get("/model-info")
def model_info():

    return {
        "type_modele": type(model).__name__,
        "nombre_arbres": model.n_estimators,
        "classes": list(model.classes_),
        "nombre_features": len(feature_cols)
    }

# Prompt systeme du LLM

SYSTEM_PROMPT = """
Tu es un assistant medical senegalais.

Tu recois un diagnostic et des donnees patient.

Explique le resultat en francais simple,
comme un medecin parlerait a son patient.

Sois rassurant mais recommande toujours
une consultation medicale.

Maximum 3 phrases.

Ne fais JAMAIS de diagnostic toi-meme.

Tu expliques uniquement le diagnostic fourni.
"""


# Route /explain

@app.post(
    "/explain",
    response_model=ExplainOutput
)

def explain(data: ExplainInput):

    """
    Expliquer un diagnostic
    en francais avec un LLM.
    """

    # Si Groq absent
    if not groq_client:

        return ExplainOutput(

            explication=
            "Service d'explication indisponible. "
            "Cle API non configuree.",

            modele_llm="aucun"
        )

    # Construire le prompt
    user_prompt = (

        f"Patient : {data.sexe}, "
        f"{data.age} ans, "
        f"region {data.region}\n"

        f"Temperature : "
        f"{data.temperature} C\n"

        f"Diagnostic du modele : "
        f"{data.diagnostic} "
        f"(probabilite "
        f"{data.probabilite:.0%})\n"

        f"Explique ce resultat "
        f"au patient."
    )

    try:

        # Appel Groq
        response = (
            groq_client.chat
            .completions.create(

                model=
                "llama-3.1-8b-instant",

                messages=[

                    {
                        "role": "system",

                        "content":
                        SYSTEM_PROMPT
                    },

                    {
                        "role": "user",

                        "content":
                        user_prompt
                    }
                ],

                max_tokens=200,

                temperature=0.3
            )
        )

        explication = (
            response
            .choices[0]
            .message.content
        )

    except Exception as e:

        explication = (
            f"Erreur lors de "
            f"de l'appel au LLM : "
            f"{str(e)}"
        )

    # Retour resultat
    return ExplainOutput(

        explication=explication
    )