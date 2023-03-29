import pytask
import matplotlib.pyplot as plt
import os
import pandas as pd




from task_upload_clean import task_load_data
@pytask.mark.task
@pytask.mark.depends_on(task_load_data, produces='data_pickle.pkl"')
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

