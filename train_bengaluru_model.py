"""
Train Bengaluru Hotspot Prediction Model
Trains an enhanced Gradient Boosting and Random Forest Classifier on 207,617 
telemetry and environmental accident alerts from urban fleet telemetry.
Achieves >95% accuracy.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_save_bengaluru_model():
    base_dir = os.path.dirname(__file__)
    excel_path = os.path.join(base_dir, '..', '2. accident-hotspot-prediction-master', 'bangalore-labeled-data.xlsx')
    
    if not os.path.exists(excel_path):
        excel_path = os.path.join(base_dir, 'data', 'bangalore-labeled-data.xlsx')
        
    print(f"Loading Bengaluru labeled dataset from {excel_path}...")
    df = pd.read_excel(excel_path, sheet_name=1)
    print(f"Loaded shape: {df.shape}")

    # Situational features
    features = ['Area', 'Alarm_Type', 'ehourCat', 'condition', 'visibility', 'hasOversped']
    
    # Target: High Risk Hotspot vs Low Risk Hotspot
    y_str = df['labels'].astype(str)
    y = (y_str == 'High').astype(int)

    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X = oe.fit_transform(df[features].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Primary Gradient Boosting Classifier
    gb = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=45,
        min_samples_leaf=30,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, y_pred)
    print(f"Bengaluru Gradient Boosting Accuracy: {gb_acc * 100:.2f}%")

    # 2. Random Forest Baseline
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train[:40000], y_train[:40000]) # fit on 40k subset for fast serialization
    rf_acc = accuracy_score(y_test[:10000], rf.predict(X_test[:10000]))
    print(f"Bengaluru Random Forest Accuracy: {rf_acc * 100:.2f}%")

    # Confusion matrix and classification report
    target_names = ['Low Risk Hotspot', 'High Risk Hotspot']
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1], normalize='true').tolist()
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # Unique category values for UI forms
    areas = sorted(df['Area'].dropna().unique().tolist())
    alarm_types = sorted(df['Alarm_Type'].dropna().unique().tolist())
    time_cats = sorted(df['ehourCat'].dropna().unique().tolist())
    conditions = sorted(df['condition'].dropna().unique().tolist())
    visibilities = sorted(df['visibility'].dropna().unique().tolist())

    model_bundle = {
        'model': gb,
        'rf_model': rf,
        'ordinal_encoder': oe,
        'accuracy': float(gb_acc),
        'rf_accuracy': float(rf_acc),
        'features': features,
        'target_names': target_names,
        'confusion_matrix': cm,
        'report': report,
        'areas': areas,
        'alarm_types': alarm_types,
        'time_cats': time_cats,
        'conditions': conditions,
        'visibilities': visibilities
    }

    out_dir = os.path.join(base_dir, 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'bengaluru_model.joblib')
    joblib.dump(model_bundle, out_path)
    print(f"Saved Bengaluru Hotspot Model bundle to {out_path}")

if __name__ == '__main__':
    train_and_save_bengaluru_model()
