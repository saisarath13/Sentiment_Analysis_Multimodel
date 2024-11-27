from flask import Flask, render_template, request
import pickle
from src.preprocess import preprocess_text  # Importing preprocess function

app = Flask(__name__)

# Load the models and vectorizer
models = {
    "SVM": pickle.load(open('models/svm_model.pkl', 'rb')),
    "Logistic Regression": pickle.load(open('models/logreg_model.pkl', 'rb')),
    "Random Forest": pickle.load(open('models/rf_model.pkl', 'rb')),
    "XGBoost": pickle.load(open('models/xgb_model.pkl', 'rb'))
}
tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))  # TF-IDF Vectorizer

# Prediction function
def get_prediction(text, model):
    # Preprocess the input text
    processed_text = preprocess_text(text)

    # Convert text into features using the loaded vectorizer (returns sparse matrix)
    features = tfidf_vectorizer.transform([processed_text])

    # Convert sparse matrix to dense matrix (important for models like SVM)
    features_dense = features.toarray()

    # Predict sentiment using the specified model
    sentiment = model.predict(features_dense)

    return sentiment[0]  # Return the predicted sentiment (0 or 1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_model = None
    text = ""

    if request.method == 'POST':
        # Get user input from the form
        text = request.form['text_input']
        selected_model = request.form['model_select']

        # Validate that a model was selected
        if not selected_model:
            return render_template(
                'index.html',
                models=list(models.keys()),
                error="Please select a model!"
            )
        
        # Use the selected model for prediction
        model = models.get(selected_model)
        if model:
            try:
                prediction = get_prediction(text, model)
            except Exception as e:
                # Handle any errors during prediction
                return render_template(
                    'index.html',
                    models=list(models.keys()),
                    error=f"An error occurred: {str(e)}"
                )

    return render_template(
        'index.html',
        models=list(models.keys()),  # Pass model names for dropdown
        selected_model=selected_model,
        text_input=text,
        prediction=prediction,
        error=None
    )

if __name__ == '__main__':
    app.run(debug=True)
