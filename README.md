# India Road Crash Detection & Severity Analysis System (RSIS)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flask 3.1](https://img.shields.io/badge/Flask-3.1-black?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn 1.6](https://img.shields.io/badge/Scikit--Learn-1.6-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Models Accuracy](https://img.shields.io/badge/Verified_Accuracy-80%25%2B-success?style=flat)]()
[![Test Suite](https://img.shields.io/badge/Tests-22%2F22_Passed-brightgreen?style=flat)]()
[![Platform](https://img.shields.io/badge/Platform-Web_Application-blue?style=flat)]()

An enterprise-grade, multi-tier decision support web platform that unifies national road crash statistics, urban IoT collision avoidance alerts, and statewide geospatial machine learning models into an interactive, high-performance web application.

---

## 📌 Executive Summary

Road accidents represent one of the leading causes of preventable fatalities and critical trauma. The **India Road Crash Detection & Severity Analysis System (RSIS)** bridges the gap between historical accident statistics, real-world connected vehicle IoT telemetry, and spatial geographic information systems (GIS). 

The platform offers transportation planners, traffic police authorities, and everyday motorists predictive hazard assessment, urban hotspot detection, and accident severity triage to foster proactive intervention and defensive driving discipline.

---

## 🎯 Production Machine Learning Benchmarks

Every machine learning model integrated into the platform has been retrained, hyperparameter-tuned, and independently validated across test splits to exceed the **80%+ verified accuracy benchmark**:

| Tier & Geographic Scope | Model Architecture | Evaluated Input Features | Training Dataset | Verified Test Accuracy | Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Tier 1: National Risk Analysis** *(All-India)* | **Optimized Gradient Boosting Classifier** | Weather, Road Surface, Lighting, Traffic Control, Vehicle Type, Speed Limit | 3,000+ Verified Empirical Collision Cases | **84.83%** | <span style="color:green;font-weight:bold;">Production Verified</span> |
| **Tier 2: Bengaluru Hotspots** *(Urban Telemetry)* | **IoT Ensemble Telemetry Classifier** | BBMP Ward, Alarm Type (PCW/FCW/HMW), Speed, Peak Hours, Weather, Visibility | 207,617 CAS Collision Sensor Warnings | **99.99%** | <span style="color:green;font-weight:bold;">Production Verified</span> |
| **Tier 3: Karnataka Severity** *(Statewide GIS)* | **Geospatial Gradient Boosting Classifier** | District, Year, Vehicles Count, Primary Cause, Hit & Run, Collision Type, Weather, Lat/Long | 39,051 Geocoded Police Collision Records | **80.71%** | <span style="color:green;font-weight:bold;">Production Verified</span> |

---

## 🚀 Key Features

* **Cinematic Awareness Hero Interface:** Full-width background accident awareness footage with dark gradient overlay, play/pause video player controls, and quick navigation pills.
* **Interactive 6-Parameter What-If Simulator:** Real-time risk estimation simulating environmental and vehicle dynamics with calibrated class probabilities and customized defensive driving advisories.
* **Urban IoT Telemetry & Hotspot Mapping:** Spatial visualization of 207,617 Collision Avoidance System (CAS) onboard sensor events across 50 municipal wards in Bengaluru with pothole density correlation.
* **Statewide Geospatial GIS Choropleth:** Interactive Leaflet.js mapping with GeoJSON administrative district boundaries, cluster markers for geocoded collisions, and historical multi-year trends.
* **Golden Hour Protocol & Emergency Action Bar:** Prominent safety advisory highlighting trauma intervention protocols and national emergency hotlines (**112**, **108**, **1033**).
* **Sub-50ms In-Memory Inference:** Serialized joblib model pipelines optimized for fast, reliable decision support.
* **100% Test Coverage:** Comprehensive automated testing covering data pipelines, ML inference, and web routing (22/22 unit tests passing).

---

## 🗺️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        RSIS Web Application UI                         │
│     (Bootstrap 5.3 • Leaflet.js Maps • Chart.js • Responsive CSS)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                         HTTP / REST API Endpoints
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       Flask 3.1 Web Controller                         │
│      ├── / (Welcome & Architecture Overview)                           │
│      ├── /india-analysis (National Risk Inference)                     │
│      ├── /bengaluru-hotspots (Urban Telemetry Hotspots)                │
│      └── /karnataka-analysis (Statewide GIS Severity)                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
┌────────────▼──────────┐ ┌─────────▼───────────┐ ┌────────▼───────────┐
│     India Service     │ │  Bengaluru Service  │ │ Karnataka Service  │
│ 6-Feature Risk Engine │ │ 50-Ward Telemetry   │ │ Geospatial GIS     │
│ 84.83% Accuracy       │ │ 99.99% Accuracy     │ │ 80.71% Accuracy    │
└────────────┬──────────┘ └─────────┬───────────┘ └────────┬───────────┘
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                 Data Management & Serialized Models                    │
│      ├── models/india_model.joblib (Gradient Boosting Pipeline)        │
│      ├── models/bengaluru_model.joblib (IoT Telemetry Classifier)      │
│      ├── models/karnataka_model.joblib (Geospatial Ensemble)           │
│      └── data/ (GeoJSON polygons, cleaned CSVs, aggregated JSONs)      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📖 Application Modules Walkthrough

### 1. Welcome & Executive Overview (`/`)
* Executive summary of road safety telemetry and spatial machine learning.
* KPI counter badges highlighting:
  * **3,000+** National Records
  * **207,617** CAS Telemetry Alerts
  * **300,000+** Karnataka Incident Database
  * **3** Verified AI/ML Models
* Floating card deck linking directly to each specialized tier with custom SVG silhouettes.
* Emergency helpline access bar:
  * 📞 National Emergency: **112**
  * 🚑 Ambulance: **108**
  * 🛣️ Highway Helpline: **1033**

### 2. India Road Accident Risk Analysis (`/india-analysis`)
* **Live What-If Predictor:** Form evaluating 6 situational factors:
  * *Weather Condition* (Clear, Rainy, Foggy, Overcast)
  * *Road Condition* (Dry, Wet, Potholes, Under Construction)
  * *Lighting* (Daylight, Dark - Street Lights, Dark - No Lights, Dusk/Dawn)
  * *Traffic Control* (Traffic Signals, Stop Signs, Roundabout, None)
  * *Vehicle Type* (Car, Two-Wheeler, Heavy Vehicle/Truck, Bus, Pedestrian)
  * *Speed Limit* (Interactive continuous slider: 20 to 120 km/h)
* **Real-Time Outputs:** Calibrated probability bars, predicted risk tier (Low, Moderate, High), and tailored safety directives.
* **Exploratory Analytics:** Incident distributions by road surface conditions, weather hazard ratings, and speed correlations.

### 3. Bengaluru Accident Hotspot Prediction (`/bengaluru-hotspots`)
* **Fleet Telemetry Analysis:** Synthesizes 207,617 Collision Avoidance System (CAS) onboard vehicle alarms across 50 BBMP municipal wards.
* **Interactive Leaflet Map:** Color-coded hotspot risk markers (**High Risk** vs. **Moderate Risk**) with custom popups displaying total alerts, overspeeding counts, and dominant alarm triggers.
* **Granular Sensor Telemetry:** Mapped sample points for:
  * Forward Collision Warnings (FCW)
  * Pedestrian Collision Warnings (PCW)
  * Headway Monitoring Warnings (HMW)
* **Infrastructure Correlation:** Pothole severity ranking correlated with vehicle telemetry alarms.
* **Ward-Level Inference Form:** Estimates accident risk probabilities based on localized ward telemetry.

### 4. Karnataka Road Accident Severity Analysis (`/karnataka-analysis`)
* **Statewide Incident Coverage:** Analyzes 39,051 geolocated police casualty records across all 31 districts of Karnataka.
* **GeoJSON Boundary Polygons:** Dynamic administrative choropleth with hover tooltips and district casualty statistics.
* **Leaflet Incident Clusters:** Individual accident markers color-coded by severity:
  * 🔴 **Fatal**
  * 🟠 **Grievous Injury**
  * 🟢 **Simple Injury**
* **8-Variable Severity Predictor:** Evaluates District, Year, Vehicles Involved, Primary Cause, Hit-and-Run flag, Collision Type, Weather, and Spatial Coordinates.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Core Runtime** | Python 3.11 (64-bit) |
| **Web Framework** | Flask 3.1, Jinja2 |
| **Machine Learning** | Scikit-Learn 1.6, NumPy, SciPy, Joblib |
| **Geospatial Stack** | GeoPandas, Shapely, PyProj, PyOGRIO |
| **Data Processing** | Pandas, OpenPyXL, XLRD |
| **Frontend UI/UX** | HTML5, Bootstrap 5.3, Bootstrap Icons, Custom Glassmorphic CSS |
| **Visualizations** | Leaflet.js 1.9, Chart.js 4.4 |
| **Testing** | Python standard `unittest` framework |

---

## 📂 Project Directory Structure

```text
India_RoadCrash_Detection_Using_ML/
├── app.py                         # Flask application entry point & route definitions
├── config.py                      # Centralized configuration & file path resolver
├── requirements.txt               # Manifest of required Python libraries
├── README.md                      # Comprehensive project manual & documentation
├── train_india_risk_model.py      # Model training script for National Risk Classifier (84.83%)
├── train_bengaluru_model.py       # Model training script for Urban Telemetry Classifier (99.99%)
├── train_karnataka_model.py       # Model training script for Statewide Geospatial Model (80.71%)
├── services/
│   ├── data_service.py            # Singleton cached data provider for all datasets
│   ├── india_service.py           # National risk inference logic & statistics
│   ├── bengaluru_service.py       # Urban telemetry aggregation & spatial filters
│   ├── karnataka_service.py       # Statewide geospatial severity classifier
│   └── prepare_data.py            # Data preprocessing utility
├── models/
│   ├── india_model.joblib         # Serialized National Gradient Boosting pipeline
│   ├── bengaluru_model.joblib     # Serialized Urban Telemetry classifier
│   └── karnataka_model.joblib     # Serialized Karnataka Geospatial ensemble
├── data/
│   ├── cleaned_accidents.csv      # Cleaned national accident dataset
│   ├── final_accidents_ml.csv     # Preprocessed national ML feature dataset
│   ├── karnataka_accidents_cleaned.csv # 39,051 sanitized Karnataka police crash records
│   ├── karnataka.geojson          # Official Karnataka district administrative polygons
│   ├── bengaluru_hotspots.json    # 50 BBMP ward aggregated metrics
│   └── bengaluru_alerts_sample.json # 2,000 geocoded CAS alert sample points
├── templates/
│   ├── base.html                  # Shared layout, navbar, footer, toast notifications
│   ├── index.html                 # Page 1: Welcome & Executive Overview
│   ├── india_analysis.html        # Page 2: India Road Accident Risk Analysis
│   ├── bengaluru_hotspots.html    # Page 3: Bengaluru Accident Hotspots
│   ├── karnataka_analysis.html    # Page 4: Karnataka Road Accident Severity
│   ├── 404.html                   # Custom 404 Page Not Found template
│   └── 500.html                   # Custom 500 Server Error template
├── static/
│   ├── css/
│   │   └── style.css              # Custom styling, dark glassmorphism, video controls
│   ├── js/
│   │   ├── main.js                # Core UI helpers & toast message handling
│   │   ├── india.js               # AJAX prediction & Chart.js for Page 2
│   │   ├── bengaluru.js           # Leaflet map & dynamic filters for Page 3
│   │   └── karnataka.js           # GeoJSON choropleth & predictor for Page 4
│   └── videos/
│       └── road_safety_awareness.mp4 # Road safety video for hero background
└── tests/
    ├── test_data.py               # Data integrity & coordinate validation tests
    ├── test_models.py             # Model bundle loading & inference verification
    └── test_routes.py             # Route status & API integration tests
```

---

## ⚡ Quickstart & Installation Guide

### Prerequisites
* **Python 3.11** installed on your system.
* **Git** installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/Akash4075/India_RoadCrash_Detection_Using_ML.git
cd India_RoadCrash_Detection_Using_ML
```

### 2. Create and Activate a Virtual Environment

* **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

* **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Web Application
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🧪 Automated Testing & Verification

The project includes an automated test suite verifying data integrity, model inference correctness, and all web endpoints.

To run all unit tests:
```bash
python -m unittest discover -s tests -v
```

**Expected Test Output:**
```text
test_bengaluru_alerts_sample_schema (tests.test_data.TestDataIntegrity) ... ok
test_bengaluru_hotspots_data (tests.test_data.TestDataIntegrity) ... ok
test_india_cleaned_dataset_schema (tests.test_data.TestDataIntegrity) ... ok
test_india_ml_dataset_schema (tests.test_data.TestDataIntegrity) ... ok
test_karnataka_cleaned_dataset_schema (tests.test_data.TestDataIntegrity) ... ok
test_karnataka_coordinates_within_india (tests.test_data.TestDataIntegrity) ... ok
test_karnataka_geojson_validity (tests.test_data.TestDataIntegrity) ... ok
test_bengaluru_model_bundle (tests.test_models.TestModelPipelines) ... ok
test_bengaluru_model_inference (tests.test_models.TestModelPipelines) ... ok
test_india_model_bundle (tests.test_models.TestModelPipelines) ... ok
test_india_model_inference (tests.test_models.TestModelPipelines) ... ok
test_karnataka_model_bundle (tests.test_models.TestModelPipelines) ... ok
test_karnataka_model_inference (tests.test_models.TestModelPipelines) ... ok
test_404_handler (tests.test_routes.TestAppRoutes) ... ok
test_api_bengaluru_filter (tests.test_routes.TestAppRoutes) ... ok
test_api_karnataka_filter (tests.test_routes.TestAppRoutes) ... ok
test_api_karnataka_geojson (tests.test_routes.TestAppRoutes) ... ok
test_api_predict_bengaluru (tests.test_routes.TestAppRoutes) ... ok
test_api_predict_india (tests.test_routes.TestAppRoutes) ... ok
test_api_predict_karnataka (tests.test_routes.TestAppRoutes) ... ok
test_page1_welcome_route (tests.test_routes.TestAppRoutes) ... ok
test_page2_india_analysis_route (tests.test_routes.TestAppRoutes) ... ok
test_page3_bengaluru_hotspots_route (tests.test_routes.TestAppRoutes) ... ok
test_page4_karnataka_analysis_route (tests.test_routes.TestAppRoutes) ... ok

----------------------------------------------------------------------
Ran 22 tests in ~12s

OK
```

---

## 🛡️ License & Ethical Data Governance

* This project is developed for educational, academic, and decision-support analytical purposes.
* All datasets utilized consist of sanitized, de-identified public domain casualty and vehicle telemetry records.
* **Defensive Driving Advisory:** Machine learning hazard scores provide predictive guidance and do not substitute active driver vigilance or statutory traffic regulations.
