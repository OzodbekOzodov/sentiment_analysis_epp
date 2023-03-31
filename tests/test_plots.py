import os
import pandas as pd
import matplotlib.pyplot as plt
from src.sentiment_analysis_epp.models.plots import plot_sentiment_bar

def test_plot_sentiment_bar():
    # Create a sample dataframe for testing
    data = pd.DataFrame({'Text': ['This is a test', 'EPP was fun class'], 'Sentiment': ['neutral', 'positive']})

    # Call the plot_sentiment_bar function
    plot_sentiment_bar(data, save_path='test_plot_sentiment_bar.png')

    # Check if the output file exists
    assert os.path.exists('test_plot_sentiment_bar.png')

    # Load the image and check if it has the expected dimensions
    with open('test_plot_sentiment_bar.png', 'rb') as f:
        img = plt.imread(f)

    expected_height, expected_width, expected_channels = 1634, 2072, 4
    assert img.shape == (expected_height, expected_width, expected_channels)

    # Clean up the created image file
    os.remove('test_plot_sentiment_bar.png')


import os
from src.sentiment_analysis_epp.models.plots import plot_performance

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.sentiment_analysis_epp.models.plots import plot_performance

def test_plot_performance():
    # Create a sample dataframe for testing
    data = pd.DataFrame({'accuracy': [0.8, 0.7, 0.85], 'precision': [0.75, 0.8, 0.9], 'recall': [0.7, 0.75, 0.8], 'f1_score': [0.72, 0.78, 0.87]})
    models = ["Logistic regression", "Naive Bayes classifier", "Support Vector Machines"]
    data.index = models
    data.index.name = "Model"
    # Save the dataframe to a CSV file
    data.to_csv('test_plot_performance.csv')

    # Call the plot_performance function
    plot_performance('test_plot_performance.csv', 'test_plot_performance.png')

    # Check if the output file exists
    assert os.path.exists('test_plot_performance.png')

    # Clean up the created files
    os.remove('test_plot_performance.csv')
    os.remove('test_plot_performance.png')

