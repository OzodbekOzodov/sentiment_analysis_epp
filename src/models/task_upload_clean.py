import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
import nltk
import pytask
from string import punctuation

@pytask.mark.task
@pytask.mark.depends_on("data.csv")
@pytask.mark.produces("data_pickle.pkl")
def task_load_data(depends_on, produces):
    """
    Load the data from "src/data_management/data.csv" and return it as a pandas DataFrame.
    """
    data = pd.read_csv("src/data_management/data.csv", encoding="ISO-8859-1")

    data = _clean_text(data)
    data = _bag_of_words(data)


import string
def _clean_text(data):
    """
    Preprocess a DataFrame column of text strings for sentiment analysis by converting them to lowercase, removing
    punctuation marks, and replacing periods and question marks with special tokens.

    Parameters:
        column (pandas.Series): The DataFrame column containing the text strings to preprocess.

    Returns:
        pandas.Series: The preprocessed text strings with special tokens for sentence segmentation.
    """
    # Convert the text to lowercase
    data['Text'] = data['Text'].str.lower()
    # Replace punctuation marks with special tokens
    data['Text'] = data['Text'].str.translate(str.maketrans('', '', string.punctuation))
    data['Text'] = data['Text'].str.replace(".", " <PERIOD> ")
    data['Text'] = data['Text'].str.replace("?", " <QUESTION> ")
    produces = "src/outputs/data_pickle.pcl"
    data.to_pickle(produces)

    


# data['soft_clean_text'] = clean_text_soft(data['Text'])
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

@pytask.mark.depends_on(
                        {task_load_data,
                        _clean_text}
                        )
@pytask.mark.produces()
def _bag_of_words(data):
    """
    Preprocess a text string for sentiment analysis by converting it to lowercase, removing punctuation marks
    and stop words, and tokenizing it into individual words.

    Parameters:
        text (str or pandas.Series): The text string or pandas Series to preprocess.

    Returns:
        str or pandas.Series: A preprocessed string or a pandas Series with preprocessed text.
    """
    if isinstance(data, str):
        # If the input is a string, preprocess it and return a string of tokens
        # Convert the text to lowercase
        data_pickle = text.lower()

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
        return text.apply(_bag_of_words)
    else:
        # Raise an error if the input is not a string or a pandas Series
        raise ValueError("Input must be a string or a pandas Series")
    






