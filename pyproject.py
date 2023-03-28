import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
import nltk
from pytask import task

@task
def load_data():
    """
    Load the data from "src/data_management/data.csv" and return it as a pandas DataFrame.
    """
    data = pd.read_csv("src/data_management/data.csv")
    return data

data = load_data()

@task(requires=[load_data])
def create_sentiment_plot(data):
    """Create a bar plot of the sentiment counts in the data and save it as a PNG image.

    Args:
        data (pandas.DataFrame): The data loaded from "src/data_management/data.csv".

    Returns:
        None
    """
    # create a pandas DataFrame with the sentiment counts
    sentiment_counts = pd.DataFrame(data['Sentiment'].value_counts())

    # create a bar plot of the sentiment counts
    sentiment_counts.plot(kind='bar', legend=None)

    # set the plot title and axis labels
    plt.title('Sentiment Counts')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    # create the data_management directory if it doesn't exist
    if not os.path.exists('data_management'):
        os.makedirs('data_management')

    # save the plot in the data_management directory
    file_name = "src/data_management/graph_1_sentiment_counts.png"
    if os.path.exists(file_name):
        os.remove(file_name)
    plt.savefig(file_name)

@task(requires=[load_data])
def clean_text_soft(column):
    """
    Preprocess a DataFrame column of text strings for sentiment analysis by converting them to lowercase, removing
    punctuation marks, and replacing periods and question marks with special tokens.

    Parameters:
        column (pandas.Series): The DataFrame column containing the text strings to preprocess.

    Returns:
        pandas.Series: The preprocessed text strings with special tokens for sentence segmentation.
    """
    # Convert the text to lowercase
    column = column.str.lower()

    # Replace punctuation marks with special tokens
    column = column.str.translate(str.maketrans('', '', string.punctuation))
    column = column.str.replace(".", " <PERIOD> ")
    column = column.str.replace("?", " <QUESTION> ")

    # Return the preprocessed text
    return column


data['soft_clean_text'] = clean_text_soft(data['Text'])

@task(requires=[clean_text_soft])
def bag_of_words(text):
    """
    Preprocess a text string for sentiment analysis by converting it to lowercase, removing punctuation marks
    and stop words, and tokenizing it into individual words.

    Parameters:
        text (str or pandas.Series): The text string or pandas Series to preprocess.

    Returns:
        str or pandas.Series: A preprocessed string or a pandas Series with preprocessed text.
    """
    if isinstance(text, str):
        # If the input is a string, preprocess it and return a string of tokens
        # Convert the text to lowercase
        text = text.lower()

        # Remove punctuation marks
        text = ''.join([c for c in text if c not in punctuation])

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [w for w in word_tokens if not w in stop_words]

        # Return the preprocessed text as a string of tokens
        return ' '.join(filtered_text)
    elif isinstance(text, pd.Series):
        # If the input is a pandas Series, preprocess each element of the Series and return a new Series
        return text.apply(bag_of_words)
    else:
        # Raise an error if the input is not a string or a pandas Series
        raise ValueError("Input must be a string or a pandas Series")
    
