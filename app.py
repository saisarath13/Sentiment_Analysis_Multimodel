from flask import Flask, render_template, request
from joblib import load
import os
from src.preprocess import preprocess_text  # Importing preprocess function

app = Flask(__name__)

# Define the base directory and models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Helper function to load a joblib file safely
def load_joblib_file(file_path):
    try:
        return load(file_path)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Load the models and vectorizer
models = {
    "SVM": load_joblib_file(os.path.join(MODELS_DIR, 'svm_model.pkl')),
    "Logistic Regression": load_joblib_file(os.path.join(MODELS_DIR, 'logreg_model.pkl')),
    "Random Forest": load_joblib_file(os.path.join(MODELS_DIR, 'rf_model.pkl')),
    "XGBoost": load_joblib_file(os.path.join(MODELS_DIR, 'xgb_model_cpu_compatible.pkl'))
}
tfidf_vectorizer = load_joblib_file(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))

# Check if all models and vectorizer are loaded correctly
if None in models.values() or tfidf_vectorizer is None:
    print("Error: One or more models/vectorizer could not be loaded. Please check the files.")
    exit(1)

# Prediction function
def get_prediction(text, model_name):
    try:
        # Preprocess the input text
        processed_text = preprocess_text(text)
        print(f"Processed text: {processed_text}")  # Debugging

        # Transform text into features
        features = tfidf_vectorizer.transform([processed_text])
        features_dense = features.toarray()
        print(f"Features shape: {features_dense.shape}")  # Debugging

        # Predict sentiment
        sentiment = models[model_name].predict(features_dense)
        print(f"Prediction: {sentiment}")  # Debugging
        return sentiment[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise ValueError("An error occurred during prediction. Please try again.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_model = None
    text = ""
    error = None

    if request.method == 'POST':
        # Get user input from the form
        text = request.form.get('text_input', '')
        selected_model = request.form.get('model_select', '')

        # Validate user inputs
        if not selected_model:
            error = "Please select a model!"
        elif not text.strip():
            error = "Please enter some text!"
        else:
            # Perform prediction
            try:
                prediction = get_prediction(text, selected_model)
            except ValueError as e:
                error = str(e)
            except Exception as e:
                print(f"Unexpected error: {e}")
                error = "An unexpected error occurred. Please try again."

    return render_template(
        'index.html',
        models=list(models.keys()),
        selected_model=selected_model,
        text_input=text,
        prediction=prediction,
        error=error
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
