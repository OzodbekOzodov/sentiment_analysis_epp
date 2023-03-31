import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import pytask
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SRC = Path(__file__).parent.parent.resolve()
BLD = SRC.joinpath("bld").resolve()

from sentiment_analysis_epp.models.plots import plot_sentiment_hist

@pytask.mark.depends_on(SRC / "sentiment_analysis_epp" / "data_management" / "data.csv")
@pytask.mark.produces(BLD / "python" / "results" / "sentiment_histogram.png")
def task_sentiment_histogram():
    # Load the data
    data_path = SRC / "sentiment_analysis_epp" / "data_management" / "data.csv"
    df = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Create the plot and save it
    plot_sentiment_hist(df)
    output_path = BLD / "python" / "results" / "sentiment_histogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return None
