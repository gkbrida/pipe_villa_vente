from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from joblib import load
import dill


with open("pipe_villa_vente/pipe.pkl", "rb") as f:
    pipe = dill.load(f)

app = FastAPI(title="Predicteur du prix de villa en vente")

# Définition des entrées
class InputData(BaseModel):
    superficie_m2: int
    nombre_pieces: int
    nombre_salles_bain: int
    jardin:int
    piscine: int
    parking: int
    cuisineEquipee: int
    securisee: int
    standing: int
    cite: int
    magasin: int
    acces: int
    meuble: int
    non_finition: int
    basse: int
    duplex: int
    triplex: int
    prix_moyen: float
    prix_min: float
    prix_max: float
    prix_median: float
    prix_q1: float
    prix_q3: float
    variance_prix: float
    titreFoncier :int

@app.post("/")
def predict_price(data: InputData):
    # Convertir en DataFrame avec une seule ligne
    df = pd.DataFrame([data.dict()])
    
    # Prédiction
    pred = pipe.predict(df)[0]
    return {"prediction": float(pred)}
