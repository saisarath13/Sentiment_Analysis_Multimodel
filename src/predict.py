import joblib
from src.preprocess import preprocess_text

# Load models
svm_model = joblib.load('models/svm_model.pkl')
logreg_model = joblib.load('models/logreg_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

def predict_sentiment(text):
    """
    Predict sentiment using all trained models.
    """
    cleaned_text = preprocess_text(text)
    features = tfidf.transform([cleaned_text]).toarray()

    predictions = {
        'svm': svm_model.predict(features)[0],
        'logreg': logreg_model.predict(features)[0],
        'rf': rf_model.predict(features)[0],
        'xgb': xgb_model.predict(features)[0]
    }
    return predictions
