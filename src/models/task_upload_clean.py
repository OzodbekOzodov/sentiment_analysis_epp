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
@pytask.mark.depends_on("src/data_management/data.csv")
@pytask.mark.produces("src/output/data_pickle.pcl")
def task_load_data(depends_on, produces):
    """
    Load the data from "src/data_management/data.csv" and return it as a pandas DataFrame.
    """
    data = pd.read_csv("src/data_management/data.csv", encoding="ISO-8859-1")

    data = _clean_text(data)
    data = _bag_of_words(data)
    produces = "src/outputs/data_pickle.pcl"
    data.to_pickle(produces)

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

    
import pytask
import matplotlib.pyplot as plt
import os
import pandas as pd
#from task_upload_clean import task_load_data

@pytask.mark.task
@pytask.mark.depends_on(task_load_data) #, produces='data_pickle.pkl"')
@pytask.mark.produces("graph_1_sentiment_counts.png")
def create_sentiment_plot(data, produces):
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
    #if not os.path.exists('outputs'):
    #    os.makedirs('outputs')

    # save the plot in the data_management directory
    file_name = "src/outputs/graph_1_sentiment_counts.png"
    if os.path.exists(file_name):
        os.remove(file_name)
    plt.savefig(file_name)



# data['soft_clean_text'] = clean_text_soft(data['Text'])
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

#@pytask.mark.depends_on(
#                       {task_load_data,
#                        _clean_text}
#                        )
#@pytask.mark.produces()
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
    






