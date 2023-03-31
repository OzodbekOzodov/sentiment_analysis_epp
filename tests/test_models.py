from __future__ import annotations

# import sentiment_analysis_epp

import pandas as pd 
import os
import sys

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

# Import the required functions from the 'models.py' module
from sentiment_analysis_epp.models.models import preprocess_data
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_analysis_epp.models.models import preprocess_data

def test_preprocess_data():
    # Create a small DataFrame with sample data
    data = pd.DataFrame({
        'Text': ['I love this product', 'This is terrible'],
        'Sentiment': [1, 0]
    })

    # Call the preprocess_data function
    preprocessed_data = preprocess_data(data)

    # Assert the presence of the 'bag_of_words' column
    assert 'bag_of_words' in preprocessed_data.columns

    # Assert that the length of the returned DataFrame is the same as the input
    assert len(preprocessed_data) == len(data)

    # Assert that the 'bag_of_words' column contains lists of strings
    for item in preprocessed_data['bag_of_words']:
        assert isinstance(item, list)
        assert all(isinstance(i, str) for i in item)

    # Assert that the bag of words representation is correct
    vectorizer = CountVectorizer()
    expected_bag_of_words = vectorizer.fit_transform(data['Text']).toarray()
    expected_bag_of_words = [list(map(str, words)) for words in expected_bag_of_words]

    assert all(a == b for a, b in zip(preprocessed_data['bag_of_words'], expected_bag_of_words))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentiment_analysis_epp.models.models import fit_logit_model


def test_fit_logit_model():
    # Create a small classification dataset
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Call the fit_logit_model function
    logit_model, logit_model_evaluation = fit_logit_model(X_train, y_train, X_test, y_test)

    # Assert that the returned model is an instance of LogisticRegression
    assert isinstance(logit_model, LogisticRegression)

    # Assert that the returned evaluation is a DataFrame with the correct shape
    assert isinstance(logit_model_evaluation, pd.DataFrame)
    assert logit_model_evaluation.shape == (1, 4)

    # Assert that the returned evaluation contains the correct columns
    expected_columns = ["accuracy", "precision", "recall", "f1_score"]
    assert all(col in logit_model_evaluation.columns for col in expected_columns)

    # Evaluate the model on the test set
    y_pred = logit_model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Assert that the returned evaluation is correct
    assert np.isclose(logit_model_evaluation["accuracy"].values[0], accuracy)
    assert np.isclose(logit_model_evaluation["precision"].values[0], precision)
    assert np.isclose(logit_model_evaluation["recall"].values[0], recall)
    assert np.isclose(logit_model_evaluation["f1_score"].values[0], f1)


from sklearn.naive_bayes import MultinomialNB
from sentiment_analysis_epp.models.models import fit_naive_bayes

def test_fit_naive_bayes():
    # Create a small classification dataset with non-negative values
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X = np.abs(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Call the fit_naive_bayes function
    nb_model, nb_evaluation_metrics = fit_naive_bayes(X_train, y_train, X_test, y_test)

    # Assert that the returned model is an instance of MultinomialNB
    assert isinstance(nb_model, MultinomialNB)

    # Assert that the returned evaluation is a DataFrame with the correct shape
    assert isinstance(nb_evaluation_metrics, pd.DataFrame)
    assert nb_evaluation_metrics.shape == (1, 4)

    # Assert that the returned evaluation contains the correct columns
    expected_columns = ["accuracy", "precision", "recall", "f1_score"]
    assert all(col in nb_evaluation_metrics.columns for col in expected_columns)

    # Evaluate the model on the test set
    y_pred = nb_model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Assert that the returned evaluation is correct
    assert np.isclose(nb_evaluation_metrics["accuracy"].values[0], accuracy)
    assert np.isclose(nb_evaluation_metrics["precision"].values[0], precision)
    assert np.isclose(nb_evaluation_metrics["recall"].values[0], recall)
    assert np.isclose(nb_evaluation_metrics["f1_score"].values[0], f1)


from sklearn.svm import SVC
from sentiment_analysis_epp.models.models import fit_svm

def test_fit_svm():
    # Create a small classification dataset
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Call the fit_svm function
    svm_model, svm_evaluation_metrics = fit_svm(X_train, y_train, X_test, y_test)

    # Assert that the returned model is an instance of SVC
    assert isinstance(svm_model, SVC)

    # Assert that the returned evaluation is a DataFrame with the correct shape
    assert isinstance(svm_evaluation_metrics, pd.DataFrame)
    assert svm_evaluation_metrics.shape == (1, 4)

    # Assert that the returned evaluation contains the correct columns
    expected_columns = ["accuracy", "precision", "recall", "f1_score"]
    assert all(col in svm_evaluation_metrics.columns for col in expected_columns)

    # Evaluate the model on the test set
    y_pred = svm_model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Assert that the returned evaluation is correct
    assert np.isclose(svm_evaluation_metrics["accuracy"].values[0], accuracy)
    assert np.isclose(svm_evaluation_metrics["precision"].values[0], precision)
    assert np.isclose(svm_evaluation_metrics["recall"].values[0], recall)
    assert np.isclose(svm_evaluation_metrics["f1_score"].values[0], f1)

import pycrfsuite
from sentiment_analysis_epp.models.models import fit_crf
import scipy.sparse as sp


def test_fit_crf():
    # Create a small classification dataset
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    # Convert the data into a sparse matrix
    X_sparse = sp.csr_matrix(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.2, random_state=42)

    # Call the fit_crf function
    crf_model, crf_evaluation_metrics = fit_crf(X_train, y_train, X_test, y_test)

    # Assert that the returned model is an instance of pycrfsuite.Tagger
    assert isinstance(crf_model, pycrfsuite.Tagger)

    # Assert that the returned evaluation is a DataFrame with the correct shape
    assert isinstance(crf_evaluation_metrics, pd.DataFrame)
    assert crf_evaluation_metrics.shape == (1, 4)

    # Assert that the returned evaluation contains the correct columns
    expected_columns = ["accuracy", "precision", "recall", "f1_score"]
    assert all(col in crf_evaluation_metrics.columns for col in expected_columns)

    # Evaluate the model on the test set
    y_pred = [crf_model.tag(xseq)[0] for xseq in X_test]

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Assert that the returned evaluation is correct
    assert np.isclose(crf_evaluation_metrics["accuracy"].values[0], accuracy)
    assert np.isclose(crf_evaluation_metrics["precision"].values[0], precision)
    assert np.isclose(crf_evaluation_metrics["recall"].values[0], recall)
    assert np.isclose(crf_evaluation_metrics["f1_score"].values[0], f1)
