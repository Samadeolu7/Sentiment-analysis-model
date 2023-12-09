import joblib

print("Testing the model")
# Load the saved model
model = joblib.load('sentiment_model.pkl')
print("Model loaded")
vectorizer = joblib.load('vectorizer.pkl')
print("Vectorizer loaded")

text = "I hate this movie"
print(text)
vectorized_text = vectorizer.transform([text])
prediction = model.predict(vectorized_text)[0]
print(prediction)
