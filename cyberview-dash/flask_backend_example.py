# Flask Backend Example for NIDS Dashboard
# Save this as app.py in your Flask project

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load your trained ML model
# model = pickle.load(open('model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# Example: Mock prediction function (replace with your actual model)
def predict_traffic(data):
    """
    Replace this with your actual ML model prediction
    data: dict with network traffic features
    """
    # Mock prediction - replace with: model.predict(...)
    # prediction = model.predict([list(data.values())])
    # confidence = model.predict_proba([list(data.values())])[0].max()
    
    prediction = "Normal" if np.random.random() > 0.3 else "Malicious"
    confidence = np.random.uniform(0.7, 0.99)
    
    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "timestamp": datetime.now().isoformat()
    }

# API Routes

@app.route('/api/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    try:
        data = request.json
        result = predict_traffic(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/predict/csv', methods=['POST'])
def predict_csv():
    """CSV batch prediction endpoint"""
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        
        results = []
        for _, row in df.iterrows():
            result = predict_traffic(row.to_dict())
            results.append(result)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/dashboard/stats', methods=['GET'])
def dashboard_stats():
    """Dashboard statistics endpoint"""
    # Replace with actual data from your database
    return jsonify({
        "total_packets": 45829,
        "normal_count": 38456,
        "malicious_count": 7373,
        "detection_accuracy": 96.8,
        "recent_detections": [
            {"timestamp": "2024-01-20 10:00", "normal": 450, "malicious": 23},
            {"timestamp": "2024-01-20 11:00", "normal": 520, "malicious": 31},
            {"timestamp": "2024-01-20 12:00", "normal": 480, "malicious": 18},
            {"timestamp": "2024-01-20 13:00", "normal": 510, "malicious": 42},
            {"timestamp": "2024-01-20 14:00", "normal": 490, "malicious": 28},
        ],
        "alerts": [
            {
                "id": "1",
                "timestamp": datetime.now().isoformat(),
                "source_ip": "192.168.1.105",
                "risk_level": "High",
                "message": "DDoS attack pattern detected"
            }
        ]
    })

@app.route('/api/analytics/feature-importance', methods=['GET'])
def feature_importance():
    """Feature importance endpoint"""
    # Replace with actual feature importance from your model
    return jsonify([
        {"feature": "dst_bytes", "importance": 0.18},
        {"feature": "src_bytes", "importance": 0.15},
        {"feature": "count", "importance": 0.12},
        {"feature": "srv_count", "importance": 0.10},
        {"feature": "same_srv_rate", "importance": 0.09},
        {"feature": "dst_host_srv_count", "importance": 0.08},
        {"feature": "protocol_type", "importance": 0.07},
        {"feature": "service", "importance": 0.06},
    ])

@app.route('/api/explainability/shap', methods=['POST'])
def shap_values():
    """SHAP values endpoint"""
    try:
        data = request.json
        # Replace with actual SHAP calculation
        # shap_values = explainer.shap_values(data)
        
        # Mock SHAP values
        return jsonify([
            {"feature": "dst_bytes", "impact": 0.23},
            {"feature": "src_bytes", "impact": -0.15},
            {"feature": "count", "impact": 0.18},
            {"feature": "srv_count", "impact": 0.12},
            {"feature": "same_srv_rate", "impact": -0.09},
            {"feature": "protocol_type", "impact": 0.08},
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Logs endpoint with filtering"""
    search = request.args.get('search', '')
    risk_level = request.args.get('risk_level', '')
    
    # Replace with actual database query
    logs = [
        {
            "id": "1",
            "timestamp": datetime.now().isoformat(),
            "source_ip": "192.168.1.105",
            "prediction": "Malicious",
            "confidence": 0.94,
            "risk_level": "High"
        }
    ]
    
    return jsonify(logs)

@app.route('/api/logs/export', methods=['GET'])
def export_logs():
    """Export logs as CSV"""
    # Replace with actual logs from database
    df = pd.DataFrame([
        {
            "timestamp": datetime.now().isoformat(),
            "source_ip": "192.168.1.105",
            "prediction": "Malicious",
            "confidence": 0.94,
            "risk_level": "High"
        }
    ])
    
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'nids_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
