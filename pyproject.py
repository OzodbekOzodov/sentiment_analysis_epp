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
    data = pd.read_csv("src/data_management/data.csv")
    return data


@task(requires=[load_data])
def create_sentiment_plot(data):
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