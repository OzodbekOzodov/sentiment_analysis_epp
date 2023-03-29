import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def train_test_vectorizer(bag_of_words, sentiment, train_size, random_state):
    """
    Split the data into training and test sets and create a CountVectorizer object.

    Parameters:
        bag_of_words (list or pandas.Series): A list of preprocessed tokens or a pandas Series with preprocessed text.
        sentiment (pandas.Series): A pandas Series with sentiment labels.
        train_size (float): The proportion of the data to use for training.
        random_state (int): The random state to use for splitting the data into training and test sets.

    Returns:
        tuple: A tuple containing the training and test data and the CountVectorizer object.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        bag_of_words, sentiment, train_size=train_size, random_state=random_state
    )

    # Vectorize the input data using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB


def train_naive_bayes(X_train, y_train):
    """
    Train a Naive Bayes classifier for sentiment analysis.

    Parameters:
        X_train (sparse matrix): The feature matrix for training data.
        y_train (pandas.Series): A pandas Series with sentiment labels for training data.

    Returns:
        tuple: A tuple containing the trained Naive Bayes classifier and the trained CountVectorizer.
    """
    # Vectorize the input data using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train Naive Bayes
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train_vec, y_train)

    return nb_clf, vectorizer


def evaluate_naive_bayes(nb_clf, vectorizer, X_test, y_test):
    """
    Evaluate a Naive Bayes classifier for sentiment analysis.

    Parameters:
        nb_clf (MultinomialNB): The trained Naive Bayes classifier.
        vectorizer (CountVectorizer): The trained CountVectorizer.
        X_test (list or pandas.Series): A list of preprocessed tokens or a pandas Series with preprocessed text for testing data.
        y_test (pandas.Series): A pandas Series with sentiment labels for testing data.

    Returns:
        dict: A dictionary with evaluation metrics for the Naive Bayes classifier.
    """
    # Vectorize the input data using the trained CountVectorizer
    X_test_vec = vectorizer.transform(X_test)

    # Evaluate Naive Bayes
    nb_pred = nb_clf.predict(X_test_vec)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    nb_precision = precision_score(y_test, nb_pred, average='weighted')
    nb_recall = recall_score(y_test, nb_pred, average='weighted')
    nb_f1 = f1_score(y_test, nb_pred, average='weighted')

    return {'accuracy': nb_accuracy, 'precision': nb_precision, 'recall': nb_recall, 'f1': nb_f1}
