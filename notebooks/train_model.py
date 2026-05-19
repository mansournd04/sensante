import pandas as pd
import numpy as np

# =====================================================
# Charger le dataset
# =====================================================

df = pd.read_csv("data/patients_dakar.csv")

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")

print(f"\nColonnes : {list(df.columns)}")

print(f"\nDiagnostics :")
print(df['diagnostic'].value_counts())

# =====================================================
# Préparation des données
# =====================================================

from sklearn.preprocessing import LabelEncoder

# Encoder les variables catégoriques
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir les features et la cible
feature_cols = [
    'age',
    'sexe_encoded',
    'temperature',
    'tension_sys',
    'toux',
    'fatigue',
    'maux_tete',
    'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print(f"\nFeatures : {X.shape}")
print(f"Cible : {y.shape}")

# =====================================================
# Séparation entraînement / test
# =====================================================

from sklearn.model_selection import train_test_split

# 80% entraînement, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nEntrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

# =====================================================
# Entraîner le modèle
# =====================================================

from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Entraîner le modèle
model.fit(X_train, y_train)

print("\nModèle entraîné !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# =====================================================
# Prédictions sur les données de test
# =====================================================

# Prédire sur les données de test
y_pred = model.predict(X_test)

# Comparer les 10 premières prédictions avec la réalité
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})

print("\nComparaison des prédictions :")
print(comparison)

# =====================================================
# Calcul de l'accuracy
# =====================================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy:.2%}")

# =====================================================
# Matrice de confusion et rapport
# =====================================================

from sklearn.metrics import confusion_matrix, classification_report

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

print("\nMatrice de confusion :")
print(cm)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# =====================================================
# Visualisation de la matrice de confusion
# =====================================================

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Créer le dossier figures s'il n'existe pas
os.makedirs("figures", exist_ok=True)

# Afficher la matrice de confusion
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Prédiction du modèle")
plt.ylabel("Vrai diagnostic")
plt.title("Matrice de confusion - SenSante")

plt.tight_layout()

# Sauvegarder l'image
plt.savefig("figures/confusion_matrix.png", dpi=150)

print("Figure sauvegardée dans figures/confusion_matrix.png")

# =====================================================
# Sérialisation du modèle
# =====================================================

import joblib

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sauvegarder le modèle
joblib.dump(model, "models/model.pkl")

# Vérifier la taille du fichier
size = os.path.getsize("models/model.pkl")

print(f"\nModèle sauvegardé : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# =====================================================
# Sauvegarder les encodeurs
# =====================================================

joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardés.")

# =====================================================
# Sauvegarder les encodeurs et les features
# =====================================================

# Sauvegarder les encodeurs
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")

# Sauvegarder la liste des features
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardés.")

# =====================================================
# Recharger le modèle sérialisé
# =====================================================

# Charger le modèle depuis le fichier
model_loaded = joblib.load("models/model.pkl")

# Charger les encodeurs
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"\nModèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# =====================================================
# Tester le modèle avec un nouveau patient
# =====================================================

# Nouveau patient
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

# Encoder les valeurs catégoriques
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Préparer les features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# Faire la prédiction
diagnostic = model_loaded.predict([features])[0]

# Probabilités
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print("\n--- Résultat du pré-diagnostic ---")

print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")

print(f"Diagnostic : {diagnostic}")

print(f"Probabilité : {proba_max:.1%}")

print("\nProbabilités par classe :")

for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"{classe:12s} : {proba:.1%} {bar}")