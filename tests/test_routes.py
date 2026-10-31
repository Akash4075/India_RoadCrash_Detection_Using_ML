import os
import sys
import unittest
import json

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app import app

class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_page1_welcome_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Road Safety Intelligence', response.data)
        self.assertIn(b'National Risk Modeling', response.data)

    def test_page2_india_analysis_route(self):
        response = self.client.get('/india-analysis')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'India Road Accident Risk Analysis', response.data)
        self.assertIn(b'Predict National Road Accident Severity', response.data)

    def test_page3_bengaluru_hotspots_route(self):
        response = self.client.get('/bengaluru-hotspots')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Bengaluru Accident Hotspot Prediction', response.data)
        self.assertIn(b'bengaluruMap', response.data)

    def test_page4_karnataka_analysis_route(self):
        response = self.client.get('/karnataka-analysis')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Karnataka Road Accident Severity', response.data)
        self.assertIn(b'karnatakaMap', response.data)

    def test_api_predict_india(self):
        payload = {
            'is_night': 1,
            'speed_limit': 80,
            'driver_age': 28,
            'driver_gender': 'Male'
        }
        response = self.client.post(
            '/api/predict/india',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('probabilities', data)

    def test_api_predict_karnataka(self):
        payload = {
            'district': 'Bengaluru City',
            'year': 2023,
            'vehicles': 2,
            'main_cause': 'Over speeding',
            'hit_run': 1,
            'collision_type': 'Hit pedestrian',
            'weather': 'Heavy Rain',
            'latitude': 12.9716,
            'longitude': 77.5946
        }
        response = self.client.post(
            '/api/predict/karnataka',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('probabilities', data)

    def test_api_predict_bengaluru(self):
        payload = {
            'area': 'Kadugodi',
            'alarm_type': 'PCW',
            'speed': 55.0,
            'time_cat': 'PeakM',
            'condition': 'Clear',
            'visibility': 'High'
        }
        response = self.client.post(
            '/api/predict/bengaluru',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('probabilities', data)

    def test_api_bengaluru_filter(self):
        response = self.client.get('/api/bengaluru/filter?risk=High%20Risk')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('hotspots', data)
        self.assertIn('count', data)

    def test_api_karnataka_filter(self):
        response = self.client.get('/api/karnataka/filter?severity=Fatal')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('markers', data)

    def test_api_karnataka_geojson(self):
        response = self.client.get('/api/karnataka/geojson')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('features', data)

    def test_404_handler(self):
        response = self.client.get('/non-existent-page')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'404 - Page Not Found', response.data)

if __name__ == '__main__':
    unittest.main()
