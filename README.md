# Restaurant Reviews Sentiment Analysis

This repository contains code for sentiment analysis of restaurant reviews using Natural Language Processing (NLP). It demonstrates how to preprocess text data, create a Bag of Words model, and train various classification algorithms to predict whether a review is positive or negative.

## Table of Contents
- [Getting Started](#getting-started)
- [Data Preprocessing](#data-preprocessing)
- [Creating the Bag of Words Model](#creating-the-bag-of-words-model)
- [Training Different Models](#training-different-models)
- [Predicting New Reviews](#predicting-new-reviews)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To run this code, you will need Python and the following libraries installed:

- NumPy
- Matplotlib
- pandas
- NLTK (Natural Language Toolkit)
- scikit-learn

## Data Preprocessing
The code performs the following data preprocessing steps:

- Removing punctuations and non-alphabetic characters
- Converting text to lowercase
- Stemming words (e.g., "loved" becomes "love")
- Removing common stopwords (e.g., "a," "the") but keeping "not"

## Creating the Bag of Words Model
The Bag of Words model is created using scikit-learn's CountVectorizer. It transforms text data into a sparse matrix where each row represents a review, and each column represents a unique word from all reviews. The cells contain binary values (0 or 1) indicating the presence of a word in a review.

## Training Different Models
The code demonstrates training several machine learning models for sentiment analysis, including:

Naive Bayes
- Logistic Regression
- K-Nearest Neighbors (K-NN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

Performance metrics such as accuracy, precision, recall, and F1-score are calculated for evaluating each model.

## Predicting New Reviews
You can use the trained models to predict the sentiment of new reviews.

## Contributing
Contributions are welcome! If you'd like to improve this code or add new features, please fork the repository and create a pull request.

## License
This project is licensed under the MIT License.

Happy analyzing restaurant reviews!
