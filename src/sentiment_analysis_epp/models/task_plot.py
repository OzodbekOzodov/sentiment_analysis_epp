import pandas as pd
import matplotlib.pyplot as plt
import pytask
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
BLD = SRC / "bld"

from sentiment_analysis_epp.models.plots import plot_sentiment_bar

@pytask.mark.depends_on(SRC / "data_management" / "data.csv")
@pytask.mark.produces(BLD / "plots" / "sentiment_hist.png")
def task_sentiment_bar(depends_on, produces):
    # Load the data
    data = pd.read_csv(depends_on, encoding="ISO-8859-1")

    # Create the histogram plot
    plot_sentiment_bar(data, save_path=produces)

    return None

import pandas as pd
import matplotlib.pyplot as plt
import pytask
from pathlib import Path
import datetime

SRC = Path(__file__).resolve().parent.parent
BLD = SRC / "bld"

from sentiment_analysis_epp.models.plots import plot_performance

@pytask.mark.depends_on(SRC / "bld" / "python" / "evaluation_metrics.csv")
@pytask.mark.produces(BLD / "plots" / f"performance_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
def task_performance_plot(depends_on, produces):
    # Call the plot_performance function
    plot_performance(input_file=depends_on, output_file=produces)

    # Clear the current figure to avoid overlapping with other plots
    plt.clf()

    return None
