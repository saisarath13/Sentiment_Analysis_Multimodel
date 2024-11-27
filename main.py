import argparse
from src.train import train_and_save_models
from src.predict import predict_sentiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], help='Mode: train or predict')
    parser.add_argument('--text', type=str, help='Text to predict sentiment (only required for predict mode)')
    args = parser.parse_args()

    if args.mode == 'train':
        # Load and preprocess your dataset
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split

        data = pd.read_csv('data/dataset.csv')  # Ensure your dataset is properly formatted
        X = data['text']
        y = data['sentiment']

        # Preprocess and vectorize
        X_cleaned = X.apply(preprocess_text)
        tfidf = TfidfVectorizer(max_features=5000)
        X_tfidf = tfidf.fit_transform(X_cleaned).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        # Save TF-IDF vectorizer
        joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

        # Train and save models
        train_and_save_models(X_train, y_train)

    elif args.mode == 'predict':
        if not args.text:
            raise ValueError("Please provide text to predict using --text argument.")
        predictions = predict_sentiment(args.text)
        print("Predictions:", predictions)

if __name__ == '__main__':
    main()
