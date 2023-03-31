import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def preprocess_data(data):
    # Instantiate a CountVectorizer object
    vectorizer = CountVectorizer()

    # Convert the 'Text' column to bag of words representation
    bag_of_words = vectorizer.fit_transform(data['Text'])

    # Store the result in a new column 'bag_of_words' as a list of strings
    data['bag_of_words'] = [list(map(str, words)) for words in bag_of_words.toarray()]

    # Convert all integers to strings in the 'bag_of_words' column
    data['bag_of_words'] = data['bag_of_words'].apply(lambda x: [str(item) for item in x])

    # Save the preprocessed data as a pickle file
    with open('src/sentiment_analysis_epp/data_management/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    return data

from sklearn.metrics import confusion_matrix

def fit_logit_model(X_train, y_train, X_test, y_test):
    # Fit the logistic regression model
    logit_model = LogisticRegression()
    logit_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = logit_model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    from sklearn.metrics import confusion_matrix

    def confusion_matrix_with_labels(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=["negative", "neutral", "positive"])
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        row_sums = np.asarray(cm_df.sum(axis=1))
        cm_df_percent = cm_df / row_sums[:, np.newaxis]
        return cm_df_percent
    conf_matrix_logit = confusion_matrix_with_labels(y_test, y_pred, labels=["negative", "neutral", "positive"])

    # Store the evaluation metrics in a dictionary
    logit_evaluation_metrics = pd.DataFrame.from_dict({
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1]
    })

    # Store the confusion matrix in a separate DataFrame
    conf_matrix_logit = pd.DataFrame(conf_matrix_logit)

    return logit_model, logit_evaluation_metrics, conf_matrix_logit




import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
def fit_naive_bayes(X_train, y_train, X_test, y_test):
    # Fit the Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    from sklearn.metrics import confusion_matrix

    def confusion_matrix_with_labels(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=["negative", "neutral", "positive"])
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        row_sums = np.asarray(cm_df.sum(axis=1))
        cm_df_percent = cm_df / row_sums[:, np.newaxis]
        return cm_df_percent
    conf_matrix_nb = confusion_matrix_with_labels(y_test, y_pred, labels=["negative", "neutral", "positive"])

    # Create a pandas DataFrame with the evaluation metrics
    nb_evaluation_metrics = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1]
    })
    # Store the confusion matrix in a separate DataFrame
    conf_matrix_nb= pd.DataFrame(conf_matrix_nb)
    return nb_model, nb_evaluation_metrics, conf_matrix_nb

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

def fit_svm(X_train, y_train, X_test, y_test):
    """Fit a Support Vector Machine classifier to data.

    Args:
        X_train (pandas.DataFrame): The training data set.
        y_train (pandas.Series): The training target labels.
        X_test (pandas.DataFrame): The test data set.
        y_test (pandas.Series): The test target labels.

    Returns:
        tuple: The fitted model (sklearn.svm.SVC) and a pandas DataFrame containing the accuracy, precision, recall, and F1 scores.
    """
    # Fit the Support Vector Machine classifier
    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    from sklearn.metrics import confusion_matrix

    def confusion_matrix_with_labels(y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=["negative", "neutral", "positive"])
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        row_sums = np.asarray(cm_df.sum(axis=1))
        cm_df_percent = cm_df / row_sums[:, np.newaxis]
        return cm_df_percent

    conf_matrix_svm = confusion_matrix_with_labels(y_test, y_pred, labels=["negative", "neutral", "positive"])
    # Create a pandas DataFrame with the evaluation metrics
    svm_evaluation_metrics = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1]
    })
    # Store the confusion matrix in a separate DataFrame
    conf_matrix_svm = pd.DataFrame(conf_matrix_svm)
    return svm_model, svm_evaluation_metrics, conf_matrix_svm


import os
import pandas as pd

def evaluation_metrics(logit_evaluation_metrics, nb_evaluation_metrics, svm_evaluation_metrics, output_csv_path):
    # Concatenate the evaluation metrics DataFrames
    combined_evaluation_metrics = pd.concat(
        [logit_evaluation_metrics, nb_evaluation_metrics, svm_evaluation_metrics],
        axis=0
    )

    # Set the index to model names
    combined_evaluation_metrics.index = ["logit", "naive_bayes", "svm"]

    # Save the combined evaluation metrics as a CSV file
    combined_evaluation_metrics.to_csv(output_csv_path, index_label="Model")

    # Convert the combined evaluation metrics to LaTeX format
    latex_table = combined_evaluation_metrics.style.to_latex()

    return latex_table

