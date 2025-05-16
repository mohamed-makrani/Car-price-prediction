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
   `git clone https://github.com/your-username/car-price-prediction.git`

2. Run the app  
   `python app.py`

## 📸 Screenshot

(Add an image of your web interface)

## 📁 Project Structure

ML-Project-Price-Detection/
│
├── cleaning_model/                  # Scripts and saved models from training
│   ├── Model.ipynb
│   ├── model_rf.joblib
│   └── model_xgb.joblib
│
├── dataset/                         # CSV datasets used for training/testing
│   ├── avito_voitures 18.csv
│   └── avito_voitures.csv
│
├── interface/                       # Flask web interface
│   ├── static/                      # Static files (e.g., CSS, JS)
│   ├── templates/
│   │   └── index.html               # HTML template for the web UI
│   └── app.py                       # Main Flask app
│
├── labelEncoder.joblib              # LabelEncoder saved for predictions
├── model_pipeline.pkl               # Full ML pipeline (encoder, scaler, model)
├── model_rf.joblib                  # Random Forest model
├── model_tree.joblib                # Decision Tree model
├── model_xgb.joblib                 # XGBoost model
├── scaler.joblib                    # Scaler used for preprocessing
│
├── web scrapping/                   # Scripts related to data scraping
│   ├── scrap.ipynb
│   └── last_scraped_page.txt
│
└── README.md                        # Project overview and instructions




