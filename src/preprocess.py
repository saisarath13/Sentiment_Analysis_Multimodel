import spacy

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """
    Preprocess text by lowercasing, removing non-alphabetic tokens, and lemmatizing.
    """
    text = text.lower()
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.text.isalpha()]
    return ' '.join(words)
