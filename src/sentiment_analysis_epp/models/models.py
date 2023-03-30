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

    # Store the result in a new column 'bag_of_words'
    data['bag_of_words'] = bag_of_words.toarray().tolist()

    # Save the preprocessed data as a pickle file
    with open('src/sentiment_analysis_epp/data_management/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    return data

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

    # Store the evaluation metrics in a dictionary
    logit_model_evaluation = pd.DataFrame.from_dict({
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1]
    })

    return logit_model, logit_model_evaluation



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

    # Create a pandas DataFrame with the evaluation metrics
    nb_evaluation_metrics = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1]
    })

    return nb_model, nb_evaluation_metrics

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

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

    # Create a pandas DataFrame with the evaluation metrics
    svm_evaluation_metrics = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1]
    })

    return svm_model, svm_evaluation_metrics


""" Second part of this script contains the codes for unsupervised ML models which does not use sentiment labels"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def lda_topic_modeling(data, n_topics=5, n_keywords=10, random_state=42):
    """
    Perform LDA topic modeling on a dataset with a "bag_of_words" column and assign topics.
    
    Args:
        data (pd.DataFrame): A DataFrame containing a "bag_of_words" column with clean text data.
        n_topics (int): The number of topics to be identified by the LDA model.
        n_keywords (int): The number of top keywords to display for each topic.
        random_state (int): The random state value for reproducibility.
        
    Returns:
        data_with_topics (pd.DataFrame): A DataFrame with the original data and an additional "topics" column containing the assigned topics.
        topic_examples (pd.DataFrame): A DataFrame containing one example headline from each topic class along with its sentiment and topic assignment.
    """
    # Create a CountVectorizer object
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(data['bag_of_words'])

    # Fit the LDA model
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    lda_model.fit(doc_term_matrix)

    # Assign topics to the headlines
    data_with_topics = data.copy()
    data_with_topics['topics'] = lda_model.transform(doc_term_matrix).argmax(axis=1)

    # Collect one example for each topic
    topic_examples = data_with_topics.groupby('topics').first().reset_index()[['Text', 'Sentiment', 'topics']]

    return data_with_topics, topic_examples


