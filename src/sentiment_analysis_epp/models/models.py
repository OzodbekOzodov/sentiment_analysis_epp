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

""" Conditional random field """

import pycrfsuite
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def fit_crf(X_train, y_train, X_test, y_test):
    def prepare_data_for_crf(X, y):
        X_crf = X.apply(lambda x: [x.to_dict()], axis=1).tolist()
        y_crf = y.apply(lambda x: [x]).tolist()
        return X_crf, y_crf

    # Prepare the data for CRF
    X_train_crf, y_train_crf = prepare_data_for_crf(X_train, y_train)
    X_test_crf, y_test_crf = prepare_data_for_crf(X_test, y_test)

    # Fit the CRF model
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train_crf, y_train_crf):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50  # stop earlier
    })

    trainer.train('crf_model.crfsuite')

    # Evaluate the model on the test set
    tagger = pycrfsuite.Tagger()
    tagger.open('crf_model.crfsuite')
    y_pred = [tagger.tag(xseq)[0] for xseq in X_test_crf]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Create a pandas DataFrame with the evaluation metrics
    crf_evaluation_metrics = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1]
    })

    return tagger, crf_evaluation_metrics



""" Second part of this script contains the codes for unsupervised ML models which does not use sentiment labels"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np 

def lda_topic_modeling(data, n_topics=20, n_keywords=20):
    """
    Perform LDA topic modeling on the given data.

    Args:
    data (pd.DataFrame): A DataFrame containing the preprocessed data with a 'bag_of_words' column.
    n_topics (int): The number of topics to generate.
    n_keywords (int): The number of keywords to display for each topic.

    Returns:
    data_with_topics (pd.DataFrame): A DataFrame with an additional 'Topic' column containing the topic number.
    topic_examples (pd.DataFrame): A DataFrame containing one example headline from each topic.
    """

    # Convert the list of words to a single string
    data['bag_of_words'] = data['bag_of_words'].apply(lambda x: ' '.join(str(word) for word in x))

    # Initialize a CountVectorizer object with the previously processed bag_of_words data
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    bag_of_words = vectorizer.fit_transform(data['bag_of_words'])

    # Perform LDA topic modeling
    lda = LatentDirichletAllocation(n_components=n_topics)
    lda.fit(bag_of_words)

    # Get the topic keywords
    keywords = np.array(vectorizer.get_feature_names_out())
    topic_keywords = []
    for topic_weights in lda.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_keywords]
        topic_keywords.append(keywords.take(top_keyword_locs).tolist())

    # Assign a topic to each document
    topic_values = lda.transform(bag_of_words)
    data['Topic'] = topic_values.argmax(axis=1)

    # Get one example headline from each topic
    topic_examples = data.groupby('Topic').first()

    return data, topic_examples
