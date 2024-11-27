# Sentiment Analysis Using Multiple Machine Learning Models

This project implements sentiment analysis using various machine learning models including Support Vector Machine (SVM), Logistic Regression, Random Forest, and XGBoost. The models are trained on text data to predict sentiment (positive or negative) from product reviews.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models & Tuning](#models--tuning)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project performs sentiment analysis on a set of product reviews. The sentiment of each review is labeled as either **positive** or **negative**. The text data is preprocessed using SpaCy for lemmatization, then transformed into TF-IDF features. The following machine learning models are implemented:

- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

Each model is fine-tuned using **GridSearchCV** to find the best hyperparameters and improve performance. The models are then saved and can be used for prediction in a real-world scenario.

## Installation

To set up this project locally, follow these steps:

1. Clone the repository using the following command:
   `git clone https://github.com/saisarath13/Sentiment_Analysis_Multimodel.git`
   Then, navigate to the project folder:
   `cd Sentiment_Analysis_Multimodel`

2. Create a virtual environment and activate it:
   - Run `python -m venv venv`
   - For activating the environment, use:
     - On Windows: `venv\Scripts\activate`
     - On macOS/Linux: `source venv/bin/activate`

3. Install the necessary dependencies with:
   `pip install -r requirements.txt`

4. Download the SpaCy model for preprocessing with:
   `python -m spacy download en_core_web_sm`

## Usage

After setting up the environment and installing the dependencies, you can run the Jupyter notebook to see the complete implementation or use the Python script to directly execute the sentiment analysis.

### Running the Sentiment Analysis

To train and evaluate the models, run the following command:

`python main.py`

This will:
- Train models with hyperparameter tuning.
- Save the models to disk.
- Evaluate model performance using metrics like accuracy, confusion matrix, and classification report.

### Predict Sentiment

To make predictions, you can call the `predict_sentiment()` function by providing a review text as input. For example:

text_input = 'I really enjoy using this product, it's fantastic!' predictions = predict_sentiment(text_input) print(predictions)


## Project Structure

The project has the following structure:

- **Sentiment_Analysis_Multimodel/**:
  - **app/**: Web application files (if any)
    - `__init__.py`
    - `routes.py`
  - **models/**: Trained models and vectorizers
    - `svm_model.pkl`
    - `logreg_model.pkl`
    - `rf_model.pkl`
    - `xgb_model.pkl`
    - `tfidf_vectorizer.pkl`
  - **notebooks/**: Jupyter Notebooks (optional for visualization)
    - `sentiment_analysis.ipynb`
  - **src/**: Source code for preprocessing, training, etc.
    - `predict.py`
    - `preprocess.py`
    - `train.py`
    - `utils.py`
  - **templates/**: HTML files (if any web app is created)
    - `index.html`
  - **tests/**: Test files for prediction and routes
    - `test_predictions.py`
    - `test_routes.py`
  - **.gitignore**: Git ignore file
  - **main.py**: Main script to run the project
  - **requirements.txt**: Required Python libraries
  - **README.md**: Project documentation
  - **sentiment_analysis.ipynb**: Jupyter notebook for analysis

## Models & Tuning

This project utilizes several machine learning models to perform sentiment analysis. Each model is fine-tuned using **GridSearchCV** to determine the optimal hyperparameters.

1. **Support Vector Machine (SVM)**
   - Hyperparameters tuned: `C`, `kernel`, `class_weight`
   - Best hyperparameters are chosen using GridSearchCV with cross-validation.

2. **Logistic Regression**
   - Hyperparameters tuned: `C`, `solver`, `max_iter`
   - Fine-tuned using GridSearchCV for optimal regularization and solver parameters.

3. **Random Forest**
   - Hyperparameters tuned: `n_estimators`, `max_depth`, `class_weight`
   - GridSearchCV used to find the best values for tree depth and class balancing.

4. **XGBoost**
   - Hyperparameters tuned: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `scale_pos_weight`
   - The XGBoost model is trained with GridSearchCV for better handling of class imbalance and overfitting.

## Results

Each model is evaluated using accuracy, confusion matrix, and classification report. The performance of each model is compared, and the best-performing model can be selected for further deployment.

### Example Output:
- **SVM Accuracy:** 0.85

#### **2. Logistic Regression**
- **Accuracy**: 0.75

#### **3. Random Forest**
- **Accuracy**: 0.85
- **Fitting process**: The model was tuned using 3 folds for each of 729 candidates, totaling 2187 fits.

#### **4. XGBoost**
- **Accuracy**: 0.75
- **Confusion Matrix**:

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or features you'd like to suggest.

### How to Contribute
1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Open a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.



