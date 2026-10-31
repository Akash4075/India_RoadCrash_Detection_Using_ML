import os
import sys
import unittest
import pandas as pd
import json

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import Config
from services.data_service import DataService

class TestDataIntegrity(unittest.TestCase):
    def setUp(self):
        self.ds = DataService.get_instance()

    def test_india_datasets_exist_and_valid(self):
        df_clean = self.ds.get_india_cleaned_data()
        df_ml = self.ds.get_india_ml_data()

        self.assertFalse(df_clean.empty, "India cleaned dataset should not be empty")
        self.assertEqual(len(df_clean), 3000, "India cleaned dataset should contain 3000 records")
        self.assertIn('Accident Severity', df_clean.columns)
        self.assertIn('Speed Limit (km/h)', df_clean.columns)

        self.assertFalse(df_ml.empty, "India ML dataset should not be empty")
        self.assertIn('Severity_Encoded', df_ml.columns)
        self.assertIn('Is_Night', df_ml.columns)

    def test_bengaluru_datasets_exist_and_valid(self):
        hotspots = self.ds.get_bengaluru_hotspots()
        alerts = self.ds.get_bengaluru_alerts()

        self.assertTrue(len(hotspots) > 0, "Bengaluru hotspots list should not be empty")
        self.assertEqual(len(hotspots), 50, "Bengaluru should contain 50 surveyed BBMP wards")
        
        # Check coordinates and bounds
        for h in hotspots:
            self.assertIn('Area', h)
            self.assertIn('Risk_Rating', h)
            self.assertTrue(12.6 <= h['lat'] <= 13.2, f"Ward {h['Area']} latitude {h['lat']} out of bounds")
            self.assertTrue(77.4 <= h['lon'] <= 77.9, f"Ward {h['Area']} longitude {h['lon']} out of bounds")

        self.assertTrue(len(alerts) > 0, "Sample CAS alerts should not be empty")

    def test_karnataka_datasets_exist_and_valid(self):
        df_k = self.ds.get_karnataka_data()
        self.assertFalse(df_k.empty, "Karnataka cleaned data should not be empty")
        self.assertTrue(len(df_k) >= 30000, f"Karnataka cleaned dataset should have >= 30k rows (got {len(df_k)})")

        required_cols = ['DISTRICTNAME', 'Severity', 'Collision_Type', 'Weather', 'Latitude', 'Longitude']
        for col in required_cols:
            self.assertIn(col, df_k.columns, f"Missing required column {col} in Karnataka data")

        # Coordinate bounds for Karnataka
        self.assertTrue((df_k['Latitude'] >= 11.0).all() and (df_k['Latitude'] <= 19.0).all(), "Invalid latitudes in Karnataka dataset")
        self.assertTrue((df_k['Longitude'] >= 73.5).all() and (df_k['Longitude'] <= 79.0).all(), "Invalid longitudes in Karnataka dataset")

        # GeoJSON validation
        geojson = self.ds.get_karnataka_geojson()
        self.assertIn('features', geojson, "GeoJSON missing features key")
        self.assertTrue(len(geojson['features']) > 0, "GeoJSON should contain district boundary features")

if __name__ == '__main__':
    unittest.main()
