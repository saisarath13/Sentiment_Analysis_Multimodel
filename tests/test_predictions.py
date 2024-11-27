import unittest
import joblib
import numpy as np
from src.predict import predict_sentiment  # Import the prediction function
from src.preprocess import preprocess_text_spacy  # Import the text preprocessing function

class TestSentimentAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load models and vectorizer before running tests"""
        # Load the saved models and TF-IDF vectorizer
        cls.svm_model = joblib.load('models/svm_model.pkl')
        cls.logreg_model = joblib.load('models/logreg_model.pkl')
        cls.rf_model = joblib.load('models/rf_model.pkl')
        cls.xgb_model = joblib.load('models/xgb_model.pkl')
        cls.tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

    def test_preprocess_text_spacy(self):
        """Test text preprocessing function"""
        text = "I really enjoy using this product, it's fantastic!"
        processed_text = preprocess_text_spacy(text)
        self.assertIsInstance(processed_text, str)
        self.assertNotIn("it's", processed_text)  # Check that 'it's' has been lemmatized.

    def test_predict_sentiment(self):
        """Test sentiment prediction function"""
        text = "I love this product!"
        predictions = predict_sentiment(text)
        
        # Ensure that predictions are in expected format
        self.assertIn('svm_prediction', predictions)
        self.assertIn('logreg_prediction', predictions)
        self.assertIn('rf_prediction', predictions)
        self.assertIn('xgb_prediction', predictions)

        # Ensure predictions are either 'positive' or 'negative'
        for model, prediction in predictions.items():
            self.assertIn(prediction, ['positive', 'negative'])

    def test_accuracy_of_models(self):
        """Check accuracy of each model"""
        # You can directly check the accuracy of the models using known test data
        test_data = ["I love this product!", "This is the worst purchase I've made."]
        expected_results = ['positive', 'negative']
        
        # Ensure the models give reasonable predictions
        for model, model_name in [
            (self.svm_model, "SVM"),
            (self.logreg_model, "Logistic Regression"),
            (self.rf_model, "Random Forest"),
            (self.xgb_model, "XGBoost")
        ]:
            with self.subTest(model=model_name):
                prediction = model.predict(self.tfidf_vectorizer.transform(test_data).toarray())
                self.assertEqual(prediction[0], expected_results[0])
                self.assertEqual(prediction[1], expected_results[1])

if __name__ == '__main__':
    unittest.main()
