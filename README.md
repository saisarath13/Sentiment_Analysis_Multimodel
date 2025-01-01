# Sentiment Analysis Multimodel

## Overview
**Sentiment Analysis Multimodel** is a web application designed to analyze and predict sentiment from textual input using multiple machine learning models, including SVM, Logistic Regression, Random Forest, and XGBoost. The application is deployed on **Google Cloud Platform (GCP)** and offers users the ability to select a model for sentiment prediction through an interactive web interface.

---

## Features
- **Multiple Models**: Compare predictions from SVM, Logistic Regression, Random Forest, and XGBoost models.
- **Interactive Web Interface**: Input text and select a model for sentiment analysis.
- **Scalable Deployment**: Deployed on GCP Container for efficient cloud-based execution.
- **Custom Preprocessing**: Utilizes SpaCy for text preprocessing and TF-IDF for feature extraction.

---

## Installation

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized deployment)
- GCP account (for deployment)

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/saisarath13/sentiment-analysis-multimodel.git
   cd sentiment-analysis-multimodel
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Access the application at `http://127.0.0.1:8080`.

---

## Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t sentiment-analysis-multimodel .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8080:8080 sentiment-analysis-multimodel
   ```

3. Access the application at `http://localhost:8080`.

---

## Deployment on GCP

1. Create a GCP project and enable Cloud Run.
2. Build and push the Docker image to Google Container Registry (GCR):
   ```bash
   gcloud builds submit --tag gcr.io/your-project-id/sentiment-analysis-multimodel
   ```
3. Deploy the image to Cloud Run:
   ```bash
   gcloud run deploy sentiment-analysis-multimodel --image gcr.io/your-project-id/sentiment-analysis-multimodel --platform managed
   ```
4. Access the deployed service using the URL provided by GCP.

---

## Project Structure
```
sentiment-analysis-multimodel/
├── app.py                # Main Flask application
├── models/               # Pre-trained models and vectorizer
│   ├── svm_model.pkl
│   ├── logreg_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   └── tfidf_vectorizer.pkl
├── templates/
│   └── index.html        # Web interface template
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── preprocess.py         # Custom text preprocessing logic
└── README.md             # Project documentation
```

---

## Usage
### Web Interface
1. Navigate to the home page.
2. Enter text in the input box.
3. Select a machine learning model from the dropdown menu.
4. Submit to view the sentiment prediction.

---

## Technologies Used
- **Flask**: Web framework for the application.
- **scikit-learn**: Machine learning models and utilities.
- **XGBoost**: Gradient boosting for advanced classification.
- **SpaCy**: Natural language processing for text preprocessing.
- **TF-IDF**: Feature extraction from text.
- **Google Cloud Platform (GCP)**: Cloud hosting.
- **Docker**: Containerization for portability.

---

## Future Enhancements
- Add more advanced NLP models like BERT or GPT for sentiment analysis.
- Improve the user interface for better user experience.
- Implement API-based usage for external integrations.
- Extend support for multilingual sentiment analysis.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For inquiries or contributions, please reach out to:
- **Email**: sarathk1307@gmail.com
- **GitHub**: https://github.com/saisarath13/

