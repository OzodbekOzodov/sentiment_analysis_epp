import pandas as pd
import matplotlib.pyplot as plt

def plot_sentiment_hist(df, save_path=None):
    # Count the number of sentiments for each class
    sentiment_counts = df["Sentiment"].value_counts()

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the histogram
    ax.bar(sentiment_counts.index, sentiment_counts.values, color=["#9b59b6", "#3498db", "#95a5a6"])

    # Add labels and titles
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Distribution of Sentiments in Reviews")

    # Save the figure if a save path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
