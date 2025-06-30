from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('readmission_model.pkl')
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    result = {
        'readmission_prediction': int(prediction[0]),
        'meaning': 'Likely to be readmitted' if prediction[0] == 1 else 'Not likely to be readmitted'
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

