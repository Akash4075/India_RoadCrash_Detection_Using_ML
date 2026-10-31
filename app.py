import logging
from flask import Flask, render_template, request, jsonify
from config import Config
from services.india_service import IndiaService
from services.bengaluru_service import BengaluruService
from services.karnataka_service import KarnatakaService
from services.data_service import DataService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize application services
data_service = DataService.get_instance()
india_service = IndiaService()
bengaluru_service = BengaluruService()
karnataka_service = KarnatakaService()

# ------------------------------------------------------------------------------
# PAGE 1: Welcome and Project Introduction
# ------------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html', active_page='home')

# ------------------------------------------------------------------------------
# PAGE 2: India Road Accident Risk Analysis
# ------------------------------------------------------------------------------
@app.route('/india-analysis')
def india_analysis():
    stats = india_service.get_summary_statistics()
    eval_metrics = india_service.get_model_evaluation()
    return render_template(
        'india_analysis.html',
        active_page='india',
        stats=stats,
        eval_metrics=eval_metrics
    )

@app.route('/api/predict/india', methods=['POST'])
def api_predict_india():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        weather = str(data.get('weather', 'Clear'))
        road_condition = str(data.get('road_condition', 'Dry'))
        lighting = str(data.get('lighting', 'Daylight'))
        traffic = str(data.get('traffic', 'Signals'))
        vehicle = str(data.get('vehicle', 'Car'))
        speed_limit = float(data.get('speed_limit', 60))

        result = india_service.predict_risk(
            weather=weather,
            road_condition=road_condition,
            lighting=lighting,
            traffic=traffic,
            vehicle=vehicle,
            speed_limit=speed_limit
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in api_predict_india: {e}")
        return jsonify({'error': str(e)}), 400

# ------------------------------------------------------------------------------
# PAGE 3: Bengaluru Accident Hotspot Prediction
# ------------------------------------------------------------------------------
@app.route('/bengaluru-hotspots')
def bengaluru_hotspots():
    stats = bengaluru_service.get_overview_statistics()
    eval_metrics = bengaluru_service.get_model_evaluation()
    initial_data = bengaluru_service.filter_hotspots()
    return render_template(
        'bengaluru_hotspots.html',
        active_page='bengaluru',
        stats=stats,
        eval_metrics=eval_metrics,
        hotspots=initial_data['hotspots'],
        sample_alerts=initial_data['sample_alerts']
    )

@app.route('/api/bengaluru/filter', methods=['GET'])
def api_bengaluru_filter():
    try:
        risk = request.args.get('risk', 'All')
        accident_sev = request.args.get('accident_severity', 'All')
        pothole_sev = request.args.get('pothole_severity', 'All')

        data = bengaluru_service.filter_hotspots(
            risk_rating=risk,
            accident_severity=accident_sev,
            pothole_severity=pothole_sev
        )
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in api_bengaluru_filter: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/bengaluru', methods=['POST'])
def api_predict_bengaluru():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        area = str(data.get('area', 'Kadugodi'))
        alarm_type = str(data.get('alarm_type', 'PCW'))
        speed = float(data.get('speed', 45))
        time_cat = str(data.get('time_cat', 'PeakM'))
        condition = str(data.get('condition', 'Clear'))
        visibility = str(data.get('visibility', 'High'))

        result = bengaluru_service.predict_hotspot(
            area=area,
            alarm_type=alarm_type,
            speed=speed,
            time_cat=time_cat,
            condition=condition,
            visibility=visibility
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in api_predict_bengaluru: {e}")
        return jsonify({'error': str(e)}), 400

# ------------------------------------------------------------------------------
# PAGE 4: Karnataka Road Accident Severity Analysis
# ------------------------------------------------------------------------------
@app.route('/karnataka-analysis')
def karnataka_analysis():
    stats = karnataka_service.get_statewide_statistics()
    eval_metrics = karnataka_service.get_model_evaluation()
    initial_markers = karnataka_service.get_map_sample(max_markers=1000)

    return render_template(
        'karnataka_analysis.html',
        active_page='karnataka',
        stats=stats,
        eval_metrics=eval_metrics,
        markers=initial_markers
    )

@app.route('/api/karnataka/geojson', methods=['GET'])
def api_karnataka_geojson():
    geojson = data_service.get_karnataka_geojson()
    return jsonify(geojson)

@app.route('/api/karnataka/filter', methods=['GET'])
def api_karnataka_filter():
    try:
        district = request.args.get('district', 'All')
        severity = request.args.get('severity', 'All')
        year = request.args.get('year', 'All')

        markers = karnataka_service.get_map_sample(
            district=district,
            severity=severity,
            year=year,
            max_markers=1500
        )
        return jsonify({'markers': markers, 'count': len(markers)})
    except Exception as e:
        logger.error(f"Error in api_karnataka_filter: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/karnataka', methods=['POST'])
def api_predict_karnataka():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        district = str(data.get('district', 'Bengaluru City'))
        year = int(data.get('year', 2023))
        vehicles = int(data.get('vehicles', 2))
        main_cause = str(data.get('main_cause', 'Over speeding'))
        hit_run = int(data.get('hit_run', 0))
        collision_type = str(data.get('collision_type', 'Vehicle to Vehicle'))
        weather = str(data.get('weather', 'Clear'))
        latitude = float(data.get('latitude', 12.9716))
        longitude = float(data.get('longitude', 77.5946))

        result = karnataka_service.predict_severity(
            district=district,
            year=year,
            vehicles=vehicles,
            main_cause=main_cause,
            hit_run=hit_run,
            collision_type=collision_type,
            weather=weather,
            latitude=latitude,
            longitude=longitude
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in api_predict_karnataka: {e}")
        return jsonify({'error': str(e)}), 400

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html', active_page=''), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html', active_page=''), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
