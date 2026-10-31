"""
Script to prepare and verify all processed working data for the Road Safety Intelligence System.
"""
import os
import json
import pandas as pd
import numpy as np

def prepare_bengaluru():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace = os.path.dirname(base_dir)
    r2 = os.path.join(workspace, '2. accident-hotspot-prediction-master')
    out_data = os.path.join(base_dir, 'data')

    cas_path = os.path.join(r2, 'bangalore-cas-alerts.xlsx')
    zones_path = os.path.join(r2, 'bangalore-accident-zones.xlsx')
    labeled_path = os.path.join(r2, 'bangalore-labeled-data.xlsx')

    print("Loading Bengaluru raw spreadsheets...")
    cas = pd.read_excel(cas_path)
    zones = pd.read_excel(zones_path)
    labeled = pd.read_excel(labeled_path, sheet_name='Sheet2')

    # Aggregate CAS alerts by ward
    ward_coords = cas.groupby('deviceCode_location_wardName').agg(
        lat=('deviceCode_location_latitude', 'mean'),
        lon=('deviceCode_location_longitude', 'mean'),
        total_alerts=('deviceCode_pyld_alarmType', 'count'),
        avg_speed=('deviceCode_pyld_speed', 'mean'),
        max_speed=('deviceCode_pyld_speed', 'max')
    ).reset_index().rename(columns={'deviceCode_location_wardName': 'Area'})

    # Group labeled data by ward
    ward_labeled = labeled.groupby('Area').agg(
        high_risk_alerts=('labels', lambda x: int((x == 'High').sum())),
        low_risk_alerts=('labels', lambda x: int((x == 'Low').sum())),
        overspeed_count=('hasOversped', lambda x: int((x == 'Yes').sum())),
        top_alarm=('Alarm_Type', lambda x: str(x.mode()[0]) if len(x) > 0 else 'N/A'),
        common_condition=('condition', lambda x: str(x.mode()[0]) if len(x) > 0 else 'Clear')
    ).reset_index()

    # Merge all three
    bengaluru_hotspots = pd.merge(ward_coords, zones, on='Area', how='left')
    bengaluru_hotspots = pd.merge(bengaluru_hotspots, ward_labeled, on='Area', how='left')

    bengaluru_hotspots['Accident_Severity'] = bengaluru_hotspots['Accident_Severity'].fillna('Medium')
    bengaluru_hotspots['Pothole_Severity'] = bengaluru_hotspots['Pothole_Severity'].fillna('Medium')
    
    median_high = bengaluru_hotspots['high_risk_alerts'].median()
    bengaluru_hotspots['Risk_Rating'] = np.where(
        bengaluru_hotspots['high_risk_alerts'] > median_high, 'High Risk', 'Moderate Risk'
    )

    hotspots_json_path = os.path.join(out_data, 'bengaluru_hotspots.json')
    bengaluru_hotspots.to_json(hotspots_json_path, orient='records', indent=2)
    print(f"Saved {len(bengaluru_hotspots)} ward hotspots to: {hotspots_json_path}")

    # Find the date column safely
    date_col = [c for c in cas.columns if 'time' in c.lower() or 'date' in c.lower()][0]
    sample_points = cas.sample(n=2000, random_state=42)[[
        'deviceCode_location_latitude',
        'deviceCode_location_longitude',
        'deviceCode_location_wardName',
        'deviceCode_pyld_alarmType',
        'deviceCode_pyld_speed',
        date_col
    ]].rename(
        columns={
            'deviceCode_location_latitude': 'lat',
            'deviceCode_location_longitude': 'lon',
            'deviceCode_location_wardName': 'ward',
            'deviceCode_pyld_alarmType': 'alarm',
            'deviceCode_pyld_speed': 'speed',
            date_col: 'time'
        }
    )
    sample_points['time'] = sample_points['time'].astype(str)

    sample_json_path = os.path.join(out_data, 'bengaluru_alerts_sample.json')
    sample_points.to_json(sample_json_path, orient='records', indent=2)
    print(f"Saved {len(sample_points)} alert samples to: {sample_json_path}")

if __name__ == '__main__':
    prepare_bengaluru()
