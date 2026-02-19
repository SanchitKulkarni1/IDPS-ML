# Flask + React Integration Guide

## Setup Instructions

### 1. Flask Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install flask flask-cors pandas numpy scikit-learn

# Save your trained model
# pickle.dump(model, open('model.pkl', 'wb'))
# pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Run Flask backend
python flask_backend_example.py
# Server will run on http://localhost:5000
```

### 2. React Frontend Setup

```bash
# Install dependencies (if not already installed)
npm install

# Create environment file
cp .env.example .env.local

# Update .env.local with your Flask backend URL
# VITE_API_URL=http://localhost:5000/api

# Run React frontend
npm run dev
# Frontend will run on http://localhost:8080
```

## API Endpoints Required

Your Flask backend should implement these endpoints:

### 1. **POST /api/predict**
- Input: JSON with network traffic features
- Output: `{ prediction: string, confidence: number, timestamp: string }`

### 2. **POST /api/predict/csv**
- Input: CSV file (multipart/form-data)
- Output: Array of predictions

### 3. **GET /api/dashboard/stats**
- Output: Dashboard statistics and recent detections

### 4. **GET /api/analytics/feature-importance**
- Output: Array of feature importance scores

### 5. **POST /api/explainability/shap**
- Input: JSON with network traffic features
- Output: Array of SHAP values

### 6. **GET /api/logs**
- Query params: `search`, `risk_level`
- Output: Array of log entries### 1. **POST /api/predict**
- Input: JSON with network traffic features
- Output: `{ prediction: string, confidence: number, timestamp: string }`

### 2. **POST /api/predict/csv**
- Input: CSV file (multipart/form-data)
- Output: Array of predictions

### 3. **GET /api/dashboard/stats**
- Output: Dashboard statistics and recent detections

### 4. **GET /api/analytics/feature-importance**
- Output: Array of feature importance scores

### 5. **POST /api/explainability/shap**
- Input: JSON with network traffic features
- Output: Array of SHAP values

### 6. **GET /api/logs**
- Query params: `search`, `risk_level`
- Output: Array of log entries

### 7. **GET /api/logs/export**
- Output: CSV file download


### 7. **GET /api/logs/export**
- Output: CSV file download

## Integration Steps

1. **Replace mock model in Flask example** (`flask_backend_example.py`):
   ```python
   # Load your actual trained model
   model = pickle.load(open('your_model.pkl', 'rb'))
   scaler = pickle.load(open('your_scaler.pkl', 'rb'))
   
   def predict_traffic(data):
       # Preprocess data
       features = scaler.transform([list(data.values())])
       # Predict
       prediction = model.predict(features)[0]
       confidence = model.predict_proba(features)[0].max()
       return {
           "prediction": "Malicious" if prediction == 1 else "Normal",
           "confidence": float(confidence),
           "timestamp": datetime.now().isoformat()
       }
   ```

2. **Add database** (optional but recommended):
   - Store predictions in SQLite/PostgreSQL
   - Track logs and alerts
   - Enable historical analytics

3. **Add SHAP explainability**:
   ```python
   import shap
   explainer = shap.TreeExplainer(model)
   
   @app.route('/api/explainability/shap', methods=['POST'])
   def shap_values():
       data = request.json
       features = scaler.transform([list(data.values())])
       shap_vals = explainer.shap_values(features)[0]
       # Return feature names and SHAP values
   ```

## Testing

1. Start Flask backend: `python flask_backend_example.py`
2. Start React frontend: `npm run dev`
3. Test each page:
   - Dashboard: View stats and charts
   - Live Detection: Submit form and upload CSV
   - Traffic Analytics: View feature importance
   - Explainability: View SHAP values
   - Logs: Filter and export logs

## Deployment

### Flask Backend
- Deploy on Heroku, AWS, DigitalOcean, or Railway
- Update `VITE_API_URL` in React `.env.local` to production URL

### React Frontend
- Build: `npm run build`
- Deploy on Vercel, Netlify, or Lovable
- Set environment variable `VITE_API_URL` to your Flask API URL

## File Structure

```
your-project/
├── flask_backend/
│   ├── app.py (flask_backend_example.py)
│   ├── model.pkl
│   ├── scaler.pkl
│   └── requirements.txt
└── react_frontend/
    ├── src/
    ├── .env.local
    └── package.json
```

## Notes

- CORS is enabled in Flask example - configure properly for production
- Add authentication if needed (JWT tokens)
- Implement rate limiting for API endpoints
- Add error handling and logging
- Use environment variables for sensitive data
