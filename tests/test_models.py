import os
import sys
import unittest

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from services.india_service import IndiaService
from services.bengaluru_service import BengaluruService
from services.karnataka_service import KarnatakaService

class TestModelInference(unittest.TestCase):
    def setUp(self):
        self.india_service = IndiaService()
        self.bengaluru_service = BengaluruService()
        self.karnataka_service = KarnatakaService()

    def test_india_model_loaded(self):
        self.assertIsNotNone(self.india_service.model_bundle, "India model bundle must be loaded")
        eval_dict = self.india_service.get_model_evaluation()
        self.assertIn('accuracy', eval_dict)
        self.assertTrue(eval_dict['accuracy'] >= 80.0, f"India accuracy is {eval_dict['accuracy']}%, expected >= 80%")

    def test_india_prediction_valid_input(self):
        res = self.india_service.predict_severity(
            is_night=0,
            speed_limit=60,
            driver_age=32,
            driver_gender='Male'
        )
        self.assertNotIn('error', res)
        self.assertIn(res['prediction'], ['Minor', 'Serious', 'Fatal'])
        self.assertIn('probabilities', res)
        total_prob = sum(res['probabilities'].values())
        self.assertAlmostEqual(total_prob, 100.0, delta=1.0)
        self.assertTrue(len(res['risk_factors']) > 0)

    def test_india_prediction_invalid_input(self):
        res = self.india_service.predict_severity(
            is_night=5, # invalid
            speed_limit=-10,
            driver_age=200,
            driver_gender='Alien'
        )
        self.assertIn('error', res)

    def test_bengaluru_model_loaded(self):
        self.assertIsNotNone(self.bengaluru_service.model_bundle, "Bengaluru model bundle must be loaded")
        eval_dict = self.bengaluru_service.get_model_evaluation()
        self.assertIn('accuracy', eval_dict)
        self.assertTrue(eval_dict['accuracy'] >= 80.0, f"Bengaluru accuracy is {eval_dict['accuracy']}%, expected >= 80%")

    def test_bengaluru_prediction_valid_input(self):
        res = self.bengaluru_service.predict_hotspot(
            area='Kadugodi',
            alarm_type='PCW',
            speed=55.0,
            time_cat='PeakM',
            condition='Clear',
            visibility='High'
        )
        self.assertNotIn('error', res)
        self.assertIn(res['prediction'], ['High Risk Hotspot', 'Low Risk Hotspot'])
        self.assertIn('probabilities', res)
        self.assertTrue(len(res['suggestions']) > 0)

    def test_karnataka_model_loaded(self):
        self.assertIsNotNone(self.karnataka_service.model_bundle, "Karnataka model bundle must be loaded")
        eval_dict = self.karnataka_service.get_model_evaluation()
        self.assertIn('verified_measured_accuracy', eval_dict)
        self.assertTrue(eval_dict['verified_measured_accuracy'] >= 80.0, f"Karnataka measured accuracy is {eval_dict['verified_measured_accuracy']}%, expected >= 80%")
        self.assertTrue(eval_dict['documented_claim_accuracy'] >= 80.0, f"Karnataka baseline accuracy is {eval_dict['documented_claim_accuracy']}%, expected >= 80%")

    def test_karnataka_prediction_valid_input(self):
        res = self.karnataka_service.predict_severity(
            district='Bengaluru City',
            year=2023,
            vehicles=2,
            main_cause='Over speeding',
            hit_run=0,
            collision_type='Vehicle to Vehicle',
            weather='Clear',
            latitude=12.9716,
            longitude=77.5946
        )
        self.assertNotIn('error', res)
        self.assertIn(res['prediction'], ['Fatal', 'Grievous Injury', 'Simple Injury'])
        self.assertIn('probabilities', res)
        total_prob = sum(res['probabilities'].values())
        self.assertAlmostEqual(total_prob, 100.0, delta=1.0)

    def test_karnataka_prediction_invalid_input(self):
        res = self.karnataka_service.predict_severity(
            district='Bengaluru City',
            year='invalid_year',
            vehicles='not_a_number',
            main_cause='Over speeding',
            hit_run=0,
            collision_type='Vehicle to Vehicle',
            weather='Clear',
            latitude=12.9716,
            longitude=77.5946
        )
        self.assertIn('error', res)

if __name__ == '__main__':
    unittest.main()
