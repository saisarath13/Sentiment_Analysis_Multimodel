import pickle

def load_model(model_path):
    """
    Load a trained model from the specified path.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
def save_model(model, model_path):
    """
    Save a trained model to the specified path.
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text, vectorizer=None):
    """
    Preprocess the input text (e.g., lemmatize, remove stopwords).
    Optionally, a vectorizer can be applied to transform the text into a vector.
    """
    # Load SpaCy model for text processing
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    # Lemmatize and remove stopwords
    processed_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

    if vectorizer:
        # If a vectorizer is provided, transform the text to a vector
        return vectorizer.transform([processed_text])
    return processed_text
def predict_sentiment(models, vectorizer, text):
    """
    Given a list of models and a vectorizer, predict sentiment for the input text.
    """
    processed_text = preprocess_text(text, vectorizer)
    predictions = {}
    
    for model_name, model in models.items():
        prediction = model.predict(processed_text)
        predictions[model_name] = prediction[0]  # Assuming model outputs a single prediction per input
        
    return predictions
