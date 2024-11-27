from flask import render_template, request
from app import app
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Load models
svm_model = joblib.load('models/svm_model.pkl')
logreg_model = joblib.load('models/logreg_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')

# Load SpaCy model for text preprocessing
nlp = spacy.load('en_core_web_sm')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

def preprocess_text_spacy(text):
    text = text.lower()
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.text.isalpha()]
    return ' '.join(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['text']
        cleaned_text = preprocess_text_spacy(user_input)
        tfidf_features = tfidf.transform([cleaned_text]).toarray()

        # Predictions from all models
        svm_pred = svm_model.predict(tfidf_features)
        logreg_pred = logreg_model.predict(tfidf_features)
        rf_pred = rf_model.predict(tfidf_features)
        xgb_pred = xgb_model.predict(tfidf_features)

        # Map predictions to labels
        labels = ['negative', 'positive']
        result = {
            'svm_prediction': labels[svm_pred[0]],
            'logreg_prediction': labels[logreg_pred[0]],
            'rf_prediction': labels[rf_pred[0]],
            'xgb_prediction': labels[xgb_pred[0]]
        }

        return render_template('index.html', prediction=result, input_text=user_input)

if __name__ == '__main__':
    app.run(debug=True)
