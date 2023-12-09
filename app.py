from flask import Flask, request, jsonify,json
import joblib


app = Flask(__name__)


# Load the saved model
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/ping')
def ping():
    return {"message": "Available", "status": 200}

# create a route that manages user request and does sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json()
    text = data['text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    if prediction == 1:
        prediction = 'Positive'
    elif prediction == -1:
        prediction = 'Negative'
    else:
        prediction = 'Neutral'
    return  jsonify({
        'status': '200',
        'success': True,
        'message': 'Prediction made successfully',
        'prediction': f'{prediction}'
    })
    


if __name__ == '__main__':
    app.run()