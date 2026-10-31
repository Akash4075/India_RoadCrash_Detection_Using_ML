import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    WORKSPACE_DIR = os.path.dirname(BASE_DIR)

    # Repository directories
    REPO1_DIR = os.path.join(WORKSPACE_DIR, '1. india-road-accident-risk-analysis-main')
    REPO2_DIR = os.path.join(WORKSPACE_DIR, '2. accident-hotspot-prediction-master')
    REPO3_DIR = os.path.join(WORKSPACE_DIR, '3. Geospatial-Machine-Learning-Model-for-Karnataka-Road-Accident-Severity-2016-2023-main')

    # Working app directories
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    # Data file paths
    INDIA_CLEANED_CSV = os.path.join(DATA_DIR, 'cleaned_accidents.csv') if os.path.exists(os.path.join(DATA_DIR, 'cleaned_accidents.csv')) else os.path.join(REPO1_DIR, 'data', 'cleaned_accidents.csv')
    INDIA_ML_CSV = os.path.join(DATA_DIR, 'final_accidents_ml.csv') if os.path.exists(os.path.join(DATA_DIR, 'final_accidents_ml.csv')) else os.path.join(REPO1_DIR, 'data', 'final_accidents_ml.csv')

    BENGALURU_HOTSPOTS_JSON = os.path.join(DATA_DIR, 'bengaluru_hotspots.json')
    BENGALURU_ALERTS_SAMPLE_JSON = os.path.join(DATA_DIR, 'bengaluru_alerts_sample.json')

    KARNATAKA_CLEANED_CSV = os.path.join(DATA_DIR, 'karnataka_accidents_cleaned.csv')
    KARNATAKA_GEOJSON = os.path.join(DATA_DIR, 'karnataka.geojson')

    # Model paths
    INDIA_MODEL_PATH = os.path.join(MODELS_DIR, 'india_model.joblib')
    KARNATAKA_MODEL_PATH = os.path.join(MODELS_DIR, 'karnataka_model.joblib')
    BENGALURU_MODEL_PATH = os.path.join(MODELS_DIR, 'bengaluru_model.joblib')

    SECRET_KEY = os.environ.get('SECRET_KEY', 'road-safety-intelligence-secret-key-2026')
    DEBUG = False
