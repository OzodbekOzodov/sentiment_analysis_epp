import pandas as pd
import matplotlib.pyplot as plt
import pytask
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
BLD = SRC / "bld"

@pytask.mark.depends_on(SRC / "data_management" / "data.csv")
@pytask.mark.produces(BLD / "plots" / "sentiment_hist.png")
def task_sentiment_hist(depends_on, produces):
    # Load the data
    data = pd.read_csv(depends_on, encoding="ISO-8859-1")

    # Create the histogram plot
    # Count the number of sentiments for each class
    sentiment_counts = data["Sentiment"].value_counts()

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the histogram
    ax.bar(sentiment_counts.index, sentiment_counts.values, color=["#9b59b6", "#3498db", "#95a5a6"])

    # Add labels and titles
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Distribution of Sentiments in Reviews")

    # Save the figure
    fig.savefig(produces, dpi=300, bbox_inches="tight")

    return None


