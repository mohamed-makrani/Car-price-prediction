# 🚗 Car Price Prediction with XGBoost + Flask

This project predicts car prices using machine learning (XGBoost) and offers a simple web interface built with Flask.

## 📦 Features

- Predict car prices based on inputs (brand, model, year, fuel, etc.)
- Pretrained XGBoost regression model
- Clean UI built using Flask
- Encoders and scaler stored and reused

## 🧠 Tech Stack

- Python, Pandas, NumPy, Scikit-learn, XGBoost
- Flask for web interface
- Joblib for model serialization

## 🏁 How to Run

1. Clone the repository  
   `git@github.com:mohamed-makrani/Car-price-prediction.git`

2. Run the app  
   `python app.py`

## 📸 Screenshot
 - Home
   
  ![1](https://github.com/user-attachments/assets/7eeb804b-9e1d-49c0-bbd8-c9ce6f6ae0a1)

 - Estimation
   
   ![2](https://github.com/user-attachments/assets/6b7f06d1-393d-4a20-99aa-96462ee02fe2)

 - Contact
   
   ![3](https://github.com/user-attachments/assets/9c5b44ec-1f9c-46b3-bbe8-d681e5b2a750)


## 📁 Project Structure

ML-Project-Price-Detection/
│
├── cleaning_model/ # Scripts and saved models from training

│ ├── Model.ipynb

│ ├── model_rf.joblib

│ └── model_xgb.joblib
│
├── dataset/ # CSV datasets used for training/testing

│ ├── avito_voitures 18.csv

│ └── avito_voitures.csv
│
├── interface/ # Flask web interface

│ ├── static/ # Static files (e.g., CSS, JS)

│ ├── templates/

│ │ └── index.html # HTML template for the web UI

│ └── app.py # Main Flask app
│
├── labelEncoder.joblib # LabelEncoder saved for predictions

├── model_pipeline.pkl # Full ML pipeline (encoder, scaler, model)

├── model_rf.joblib # Random Forest model

├── model_tree.joblib # Decision Tree model

├── model_xgb.joblib # XGBoost model

├── scaler.joblib # Scaler used for preprocessing
│
├── web scrapping/ # Scripts related to data scraping

│ ├── scrap.ipynb

│ └── last_scraped_page.txt
│

└── README.md # Project overview and instructions




