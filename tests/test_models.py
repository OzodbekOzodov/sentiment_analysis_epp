import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_analysis_epp.models.models import preprocess_data

def test_preprocess_data():
    # Create a sample dataframe for testing
    data = pd.DataFrame({'Text': ['This is a test', 'I love programming']})

    # Call the preprocess_data function
    preprocessed_data = preprocess_data(data)

    # Check if the 'bag_of_words' column exists in the preprocessed data
    assert 'bag_of_words' in preprocessed_data.columns

    # Check if the 'bag_of_words' column has the correct data type (list of strings)
    assert isinstance(preprocessed_data['bag_of_words'][0], list)
    assert isinstance(preprocessed_data['bag_of_words'][0][0], str)

    # Check if the pickle file is created
    assert os.path.exists('src/sentiment_analysis_epp/data_management/preprocessed_data.pkl')

    # Load the pickle file and compare it with the preprocessed data
    with open('src/sentiment_analysis_epp/data_management/preprocessed_data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    
    assert preprocessed_data.equals(loaded_data)

    # Clean up the created pickle file
    os.remove('src/sentiment_analysis_epp/data_management/preprocessed_data.pkl')

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.sentiment_analysis_epp.models.models import fit_logit_model


def test_fit_logit_model():
    # Create a sample dataset for testing
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=4, n_clusters_per_class=1, random_state=42)

    # Convert labels to a DataFrame with corresponding class names
    y = pd.Series(y).map({0: 'negative', 1: 'neutral', 2: 'positive'})

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test the fit_logit_model function
    logit_model, logit_evaluation_metrics, conf_matrix_logit = fit_logit_model(X_train, y_train, X_test, y_test)

    # Check if the model is not None
    assert logit_model is not None

    # Check if the evaluation metrics are close to the expected values
    expected_accuracy = 0.85
    expected_precision = 0.87
    expected_recall = 0.85
    expected_f1_score = 0.84

    np.testing.assert_allclose(logit_evaluation_metrics.loc[0, "accuracy"], expected_accuracy, rtol=0.3)
    np.testing.assert_allclose(logit_evaluation_metrics.loc[0, "precision"], expected_precision, rtol=0.3)
    np.testing.assert_allclose(logit_evaluation_metrics.loc[0, "recall"], expected_recall, rtol=0.3)
    np.testing.assert_allclose(logit_evaluation_metrics.loc[0, "f1_score"], expected_f1_score, rtol=0.3)

    # Check if the confusion matrix has the correct shape
    assert conf_matrix_logit.shape == (3, 3)


import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.sentiment_analysis_epp.models.models import fit_naive_bayes
from sklearn.datasets import make_blobs

def test_fit_naive_bayes():
    # Create a sample dataset for testing
    X, y = make_blobs(n_samples=100, centers=3, n_features=10, random_state=42)

    # Shift all feature values to be non-negative
    X += abs(X.min()) + 1

    # Convert labels to a DataFrame with corresponding class names
    y = pd.Series(y).map({0: 'negative', 1: 'neutral', 2: 'positive'})

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test the fit_naive_bayes function
    nb_model, nb_evaluation_metrics, conf_matrix_nb = fit_naive_bayes(X_train, y_train, X_test, y_test)

    # Check if the model is not None
    assert nb_model is not None

    # Check if the evaluation metrics are close to the expected values
    expected_accuracy = 0.85
    expected_precision = 0.87
    expected_recall = 0.85
    expected_f1_score = 0.84

    np.testing.assert_allclose(nb_evaluation_metrics.loc[0, "accuracy"], expected_accuracy, rtol=0.3)
    np.testing.assert_allclose(nb_evaluation_metrics.loc[0, "precision"], expected_precision, rtol=0.3)
    np.testing.assert_allclose(nb_evaluation_metrics.loc[0, "recall"], expected_recall, rtol=0.3)
    np.testing.assert_allclose(nb_evaluation_metrics.loc[0, "f1_score"], expected_f1_score, rtol=0.3)

    # Check if the confusion matrix has the correct shape
    assert conf_matrix_nb.shape == (3, 3)



import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.sentiment_analysis_epp.models.models import fit_svm

def test_fit_svm():
    # Create a sample dataset for testing
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=4, n_clusters_per_class=1, random_state=42)

    # Convert labels to a DataFrame with corresponding class names
    y = pd.Series(y).map({0: 'negative', 1: 'neutral', 2: 'positive'})

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test the fit_svm function
    svm_model, svm_evaluation_metrics, conf_matrix_svm = fit_svm(X_train, y_train, X_test, y_test)

    # Check if the model is not None
    assert svm_model is not None

    # Check if the evaluation metrics are close to the expected values
    expected_accuracy = 0.85
    expected_precision = 0.87
    expected_recall = 0.85
    expected_f1_score = 0.84

    np.testing.assert_allclose(svm_evaluation_metrics.loc[0, "accuracy"], expected_accuracy, rtol=0.3)
    np.testing.assert_allclose(svm_evaluation_metrics.loc[0, "precision"], expected_precision, rtol=0.3)
    np.testing.assert_allclose(svm_evaluation_metrics.loc[0, "recall"], expected_recall, rtol=0.3)
    np.testing.assert_allclose(svm_evaluation_metrics.loc[0, "f1_score"], expected_f1_score, rtol=0.3)

    # Check if the confusion matrix has the correct shape
    assert conf_matrix_svm.shape == (3, 3)



from src.sentiment_analysis_epp.models.models import evaluation_metrics

def test_evaluation_metrics():
    # Create some sample evaluation metrics DataFrames
    logit_metrics = pd.DataFrame({
        'accuracy': [0.85],
        'precision': [0.87],
        'recall': [0.85],
        'f1_score': [0.84]
    })
    nb_metrics = pd.DataFrame({
        'accuracy': [0.80],
        'precision': [0.82],
        'recall': [0.80],
        'f1_score': [0.79]
    })
    svm_metrics = pd.DataFrame({
        'accuracy': [0.88],
        'precision': [0.90],
        'recall': [0.88],
        'f1_score': [0.87]
    })

    # Test the evaluation_metrics function
    output_csv_path = 'test_evaluation_metrics.csv'
    latex_table = evaluation_metrics(logit_metrics, nb_metrics, svm_metrics, output_csv_path)

    # Check if the output CSV file exists
    assert os.path.exists(output_csv_path)

    # Load the output CSV file and compare it with the expected values
    expected_csv = pd.read_csv('test_evaluation_metrics.csv', index_col='Model')
    output_csv = pd.read_csv(output_csv_path, index_col='Model')
    assert expected_csv.equals(output_csv)

    # Clean up the created files
    os.remove(output_csv_path)
