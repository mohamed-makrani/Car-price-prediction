import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Création de l'application Flask
app = Flask(__name__)


# Charger le modèle XGBoost sauvegardé
try:
    model = joblib.load("model_xgb.joblib")
    scaler = joblib.load("scaler.joblib")
    label_encoders = joblib.load('labelEncoder.joblib')
    print("Modèle XGBoost chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

# Charger les encodeurs (au moment de l'entraînement, vous devrez ajuster les encodeurs sur les données)

# Chargement et ajustement du scaler avec des données d'entraînement (assurez-vous que ces données sont les mêmes que celles utilisées pour l'entraînement)
# Par exemple, vous pouvez charger un fichier avec vos données d'entraînement
# scaler.fit(training_data[numerical_cols])

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
            def clean_string(s):
                return s.strip().capitalize().replace(" ", "").replace("-", "")

            marque = clean_string(request.form['Marque'])
            module = clean_string(request.form['Module'])
            auto_manuel = clean_string(request.form['Auto_manuel'])
            carburant = clean_string(request.form['Carburant'])
            eat = clean_string(request.form['Eat'])
            # Récupérer les données du formulaire
            # marque = request.form['Marque']
            # module = request.form['Module']
            kilometrage = int(request.form['Kilomtrage'])
            annee = int(request.form['Annee'])
            puissance_fiscale = int(request.form['Puissance_fiscale'])
            # auto_manuel = request.form['Auto_manuel']
            # carburant = request.form['Carburant']
            nombre_de_portes = int(request.form['Nombre_de_portes'])
            # eat = request.form['Eat']

            # Créer un dictionnaire de données
            input_data = {
                'Marque': marque,
                'Module': module,
                'Kilomtrage': kilometrage,
                'Annee': annee,
                'Puissance_fiscale': puissance_fiscale,
                'Auto_manuel': auto_manuel,
                'Carburant': carburant,
                'Nombre_de_portes': nombre_de_portes,
                'Eat': eat
            }
        
            input_df = pd.DataFrame([input_data])
            # Liste des colonnes catégorielles
            categorical_cols = ['Marque', 'Module', 'Auto_manuel', 'Carburant', 'Eat']

            # Transformation des colonnes catégorielles avec les encodeurs
            for col in categorical_cols:
                input_df[col] = label_encoders[col].transform(input_df[col])

            # Liste des colonnes numériques à standardiser
            numerical_cols = ['Kilomtrage', 'Annee', 'Puissance_fiscale', 'Nombre_de_portes']

            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

            input_X = input_df[categorical_cols + numerical_cols]
            prediction = model.predict(input_X)
            predicted_price = np.expm1(prediction[0])
            # predicted_price = prediction[0]

            # Retourner le résultat dans le template HTML
            return render_template('index.html', price=round(predicted_price, 2))

    return render_template('index.html', price=None)

if __name__ == "__main__":
    app.run(debug=True)
