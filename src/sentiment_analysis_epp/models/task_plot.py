import pytask
from pathlib import Path
import pandas as pd
from sentiment_analysis_epp.models.plots import plot_sentiment_hist

SRC = Path(__file__).resolve().parent.parent
BLD = SRC / "bld"

@pytask.mark.depends_on(SRC / "data_management" / "data.csv")
@pytask.mark.produces(BLD / "python" / "results" / "sentiment_hist.png")
def task_plot_sentiment_hist(depends_on, produces):
    data_path = depends_on
    output_path = produces

    # Read the data into a DataFrame
    df = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Call the plot_sentiment_hist() function to create the plot
    plot_sentiment_hist(df, save_path=output_path)

""" Model performance plot """
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
import pytask
from sentiment_analysis_epp.models.plots import plot_performance

SRC = Path(__file__).resolve().parent.parent
BLD = SRC / "bld"

@pytask.mark.depends_on(
    {
        SRC / "bld" / "python" / "evaluation_metrics.csv",
    })
@pytask.mark.produces(BLD / "python" / "results" / "performance_bar.png")
def task_plot_performance(depends_on, produces):
    # Define the input file containing the evaluation metrics
    input_file = depends_on[0]

    # Define the output file where the plot will be saved
    output_file = produces

    # Call the plot_performance() function to create the plot
    plot_performance(input_file, output_file)
