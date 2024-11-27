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

