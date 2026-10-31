import os
import json
import logging
import pandas as pd
from config import Config

logger = logging.getLogger(__name__)

class DataService:
    _instance = None
    _india_cleaned_df = None
    _india_ml_df = None
    _bengaluru_hotspots = None
    _bengaluru_alerts = None
    _karnataka_df = None
    _karnataka_geojson = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_india_cleaned_data(self) -> pd.DataFrame:
        if self._india_cleaned_df is None:
            if os.path.exists(Config.INDIA_CLEANED_CSV):
                self._india_cleaned_df = pd.read_csv(Config.INDIA_CLEANED_CSV)
                logger.info(f"Loaded India cleaned dataset: {len(self._india_cleaned_df)} rows")
            else:
                logger.warning(f"File not found: {Config.INDIA_CLEANED_CSV}")
                self._india_cleaned_df = pd.DataFrame()
        return self._india_cleaned_df

    def get_india_ml_data(self) -> pd.DataFrame:
        if self._india_ml_df is None:
            if os.path.exists(Config.INDIA_ML_CSV):
                self._india_ml_df = pd.read_csv(Config.INDIA_ML_CSV)
                logger.info(f"Loaded India ML dataset: {len(self._india_ml_df)} rows")
            else:
                logger.warning(f"File not found: {Config.INDIA_ML_CSV}")
                self._india_ml_df = pd.DataFrame()
        return self._india_ml_df

    def get_bengaluru_hotspots(self) -> list:
        if self._bengaluru_hotspots is None:
            if os.path.exists(Config.BENGALURU_HOTSPOTS_JSON):
                with open(Config.BENGALURU_HOTSPOTS_JSON, 'r', encoding='utf-8') as f:
                    self._bengaluru_hotspots = json.load(f)
                logger.info(f"Loaded Bengaluru hotspots: {len(self._bengaluru_hotspots)} wards")
            else:
                logger.warning(f"File not found: {Config.BENGALURU_HOTSPOTS_JSON}")
                self._bengaluru_hotspots = []
        return self._bengaluru_hotspots

    def get_bengaluru_alerts(self) -> list:
        if self._bengaluru_alerts is None:
            if os.path.exists(Config.BENGALURU_ALERTS_SAMPLE_JSON):
                with open(Config.BENGALURU_ALERTS_SAMPLE_JSON, 'r', encoding='utf-8') as f:
                    self._bengaluru_alerts = json.load(f)
                logger.info(f"Loaded Bengaluru sample alerts: {len(self._bengaluru_alerts)} events")
            else:
                logger.warning(f"File not found: {Config.BENGALURU_ALERTS_SAMPLE_JSON}")
                self._bengaluru_alerts = []
        return self._bengaluru_alerts

    def get_karnataka_data(self) -> pd.DataFrame:
        if self._karnataka_df is None:
            if os.path.exists(Config.KARNATAKA_CLEANED_CSV):
                self._karnataka_df = pd.read_csv(Config.KARNATAKA_CLEANED_CSV)
                logger.info(f"Loaded Karnataka cleaned dataset: {len(self._karnataka_df)} rows")
            else:
                logger.warning(f"File not found: {Config.KARNATAKA_CLEANED_CSV}")
                self._karnataka_df = pd.DataFrame()
        return self._karnataka_df

    def get_karnataka_geojson(self) -> dict:
        if self._karnataka_geojson is None:
            if os.path.exists(Config.KARNATAKA_GEOJSON):
                with open(Config.KARNATAKA_GEOJSON, 'r', encoding='utf-8') as f:
                    self._karnataka_geojson = json.load(f)
                logger.info("Loaded Karnataka GeoJSON district boundary polygons")
            else:
                logger.warning(f"File not found: {Config.KARNATAKA_GEOJSON}")
                self._karnataka_geojson = {"type": "FeatureCollection", "features": []}
        return self._karnataka_geojson
