import pandas as pd
import matplotlib.pyplot as plt
import sys

import sys
print("Importing module:", __name__)


def plot_sentiment_bar(df, save_path=None):

    # Count the number of sentiments for each class
    sentiment_counts = df["Sentiment"].value_counts()

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the histogram
    ax.bar(sentiment_counts.index, sentiment_counts.values, color=["#95a5a6", "#95a5a6", "#95a5a6"])

    # Add labels and titles
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Distribution of Sentiments in Reviews")

    # Save the figure if a save path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_performance(input_file, output_file):
    df = pd.read_csv(input_file)

    models = df.iloc[:, 0]
    metrics = df.iloc[:, 1:].astype(float)
    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1_score = metrics['f1_score']

    barWidth = 0.2

    r1 = np.arange(len(models))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    # Create a new figure for the second plot with transparent background
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0.0)

    ax.bar(r1, accuracy, color='gray', hatch='////', width=barWidth, edgecolor='white', label='Accuracy')
    ax.bar(r2, precision, color='gray', hatch='....', width=barWidth, edgecolor='white', label='Precision')
    ax.bar(r3, recall, color='gray', hatch='xxxx', width=barWidth, edgecolor='white', label='Recall')
    ax.bar(r4, f1_score, color='gray', hatch='++++', width=barWidth, edgecolor='white', label='F1 Score')
    ax.set_ylim([0.7, 0.88])
    ax.set_xticks([r + barWidth for r in range(len(models))])
    ax.set_xticklabels(models)
    ax.legend(ncol=4, bbox_to_anchor=(0, -0.15), loc='lower left')
    
    # Save the plot as a file
    fig.savefig(output_file, format='png', dpi=300, bbox_inches="tight", transparent=True)
