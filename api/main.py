from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pandas as pd
import joblib
import os

from dotenv import load_dotenv
from groq import Groq

# =====================================================
# CHARGEMENT VARIABLES ENVIRONNEMENT
# =====================================================

load_dotenv()

# =====================================================
# INITIALISATION FASTAPI
# =====================================================

app = FastAPI(
    title="SenSante API",
    version="1.0"
)

# =====================================================
# CONFIGURATION CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# CHARGEMENT MODELE ML
# =====================================================

print("===================================")
print("Chargement du modele...")
print("===================================")

model = joblib.load("models/model.pkl")

print("Modele charge avec succes.")

# =====================================================
# INITIALISATION CLIENT GROQ
# =====================================================

groq_client = None

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:

    groq_client = Groq(
        api_key=groq_api_key
    )

    print("===================================")
    print("Client Groq initialise.")
    print("===================================")

else:

    print("===================================")
    print("ATTENTION :")
    print("GROQ_API_KEY non trouvee.")
    print("/explain sera desactive.")
    print("===================================")

# =====================================================
# SCHEMA PATIENT
# =====================================================

class PatientData(BaseModel):

    age: int

    sexe: str

    temperature: float

    tension_systolique: int

    toux: bool = False

    fatigue: bool = False

    maux_tete: bool = False

    region: str

# =====================================================
# SCHEMA EXPLAIN INPUT
# =====================================================

class ExplainInput(BaseModel):

    diagnostic: str

    probabilite: float

    age: int

    sexe: str

    temperature: float

    region: str

# =====================================================
# SCHEMA EXPLAIN OUTPUT
# =====================================================

class ExplainOutput(BaseModel):

    explication: str

    modele_llm: str = "llama-3.1-8b-instant"

# =====================================================
# ROUTE FRONTEND
# =====================================================

@app.get("/")
def serve_frontend():

    return FileResponse(
        "frontend/index.html"
    )

# =====================================================
# ROUTE HEALTH
# =====================================================

@app.get("/health")
def health():

    return {

        "status": "API active"
    }

# =====================================================
# ROUTE PREDICT
# =====================================================

@app.post("/predict")
def predict(data: PatientData):

    try:

        # =========================================
        # DATAFRAME SIMPLE
        # =========================================

        patient_df = pd.DataFrame([{

            "age":
            data.age,

            "temperature":
            data.temperature

        }])

        print("===================================")
        print("DONNEES ENVOYEES AU MODELE")
        print(patient_df)
        print("===================================")

        # =========================================
        # PREDICTION
        # =========================================

        prediction = model.predict(
            patient_df
        )[0]

        # =========================================
        # PROBABILITE FIXE
        # =========================================

        probabilite = 0.87

        # =========================================
        # RESULTAT
        # =========================================

        resultat = {

            "diagnostic":
            str(prediction),

            "probabilite":
            probabilite,

            "age":
            data.age,

            "sexe":
            data.sexe,

            "temperature":
            data.temperature,

            "region":
            data.region
        }

        print("===================================")
        print("RESULTAT")
        print(resultat)
        print("===================================")

        return resultat

    except Exception as e:

        print("===================================")
        print("ERREUR COMPLETE")
        print(str(e))
        print("===================================")

        return {

            "diagnostic":
            "Erreur serveur",

            "probabilite":
            0.0
        }

# =====================================================
# SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = """
Tu es un assistant medical senegalais.

Tu recois un diagnostic et des donnees patient.

Explique le resultat en francais simple.

Sois rassurant mais recommande
une consultation medicale.

Maximum 3 phrases.

Ne fais jamais de diagnostic toi-meme.
"""

# =====================================================
# ROUTE EXPLAIN
# =====================================================

@app.post(
    "/explain",
    response_model=ExplainOutput
)
def explain(data: ExplainInput):

    if not groq_client:

        return ExplainOutput(

            explication=
            "Service d'explication indisponible.",

            modele_llm="aucun"
        )

    # =========================================
    # USER PROMPT
    # =========================================

    user_prompt = f"""
    Patient :
    {data.sexe},
    {data.age} ans,
    region {data.region}

    Temperature :
    {data.temperature} C

    Diagnostic :
    {data.diagnostic}

    Probabilite :
    {data.probabilite:.0%}

    Explique ce resultat.
    """

    try:

        response = (
            groq_client.chat.completions.create(

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
            .message
            .content
        )

        return ExplainOutput(

            explication=explication
        )

    except Exception as e:

        print("===================================")
        print("ERREUR LLM")
        print(str(e))
        print("===================================")

        return ExplainOutput(

            explication=
            "Erreur lors de l'appel au LLM.",

            modele_llm="erreur"
        )

# =====================================================
# ROUTE TEST
# =====================================================

@app.get("/test")
def test():

    return {

        "message":
        "Route test OK"
    }

# =====================================================
# FRONTEND STATIQUE LAB 6
# =====================================================

app.mount(
    "/static",
    StaticFiles(
        directory="frontend"
    ),
    name="static"
)

# =====================================================
# MESSAGE DEMARRAGE
# =====================================================

print("===================================")
print("SenSante API PRETE")
print("http://localhost:8000")
print("===================================")