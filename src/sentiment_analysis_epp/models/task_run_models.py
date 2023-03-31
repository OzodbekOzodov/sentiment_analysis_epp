import sys
import os
import pytask
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "bld").resolve()

# Add the src directory to the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#from sentiment_analysis_epp.src.models.models import fit_logit_model, fit_naive_bayes, fit_svm

from sentiment_analysis_epp.models.models import fit_logit_model, fit_naive_bayes, fit_svm
from sentiment_analysis_epp.models.models import fit_logit_model, fit_naive_bayes, fit_svm

import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import pytask
from sklearn.feature_extraction.text import TfidfVectorizer


SRC = Path(__file__).resolve().parent.parent
BLD = SRC / "bld"


@pytask.mark.depends_on(SRC / "data_management" / "data.csv")
@pytask.mark.produces(BLD / "python" / "data" / "preprocessed_data.pkl")
def task_preprocess_data():
    # Load the raw data
    data_path = SRC / "data_management" / "data.csv"
    data = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Instantiate a CountVectorizer object
    vectorizer = CountVectorizer()

    # Convert the 'Text' column to bag of words representation
    bag_of_words = vectorizer.fit_transform(data['Text'])

    # Store the result in a new column 'bag_of_words'
    data['bag_of_words'] = bag_of_words.toarray().tolist()

    # Save the preprocessed data to a pickle file
    output_path = BLD / "python" / "data" / "preprocessed_data.pkl"
    with open(output_path, "wb") as file:
        pickle.dump(data, file)

    return None


import pickle
import pandas as pd

@pytask.mark.depends_on(SRC / "bld" / "python" / "data" / "preprocessed_data.pkl")
@pytask.mark.produces([
    BLD / "python" / "models" / "logit_model.pkl",
    BLD / "python" / "evaluation_metrics_logit.csv",
    BLD / "python" / "conf_matrix_logit.tex"
])
def task_run_logit(depends_on, produces):
    # Load preprocessed data
    with open(depends_on, "rb") as f:
        data = pickle.load(f)

    # Vectorize the 'Text' column
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Text'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, data['Sentiment'], random_state=42)

    # Fit the logistic regression model
    logit_model, logit_evaluation, conf_matrix_logit = fit_logit_model(X_train, y_train, X_test, y_test)

    # Save the logit model
    with open(produces[0], "wb") as f:
        pickle.dump(logit_model, f)

    # Save the logit evaluation metrics
    logit_evaluation.index = ["logit"]
    logit_evaluation.to_csv(produces[1], index_label="Model")

    # Save the logit confusion matrix
    with open(produces[2], "w") as f:
        conf_matrix_logit.style.to_latex(buf=f, caption="Confusion matrix for logit model")

    return None
@pytask.mark.depends_on(SRC / "bld" / "python" / "data" / "preprocessed_data.pkl")
@pytask.mark.produces(BLD / "python" / "models" / "naive_bayes_model.pkl")
@pytask.mark.produces([
    BLD / "python" / "models" / "naive_bayes_model.pkl",
    BLD / "python" / "evaluation_metrics_nb.csv",
    BLD / "python" / "conf_matrix_nb.tex"
])
def task_run_naive_bayes(depends_on, produces):
    # Load preprocessed data
    with open(depends_on, "rb") as f:
        data = pickle.load(f)

    # Vectorize the 'Text' column
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Text'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, data['Sentiment'], random_state=42)

    # Fit the model
    nb_model, nb_evaluation_metrics, conf_matrix_nb = fit_naive_bayes(X_train, y_train, X_test, y_test)

    # Save the model
    with open(produces, "wb") as f:
        pickle.dump(nb_model, f)

    # Save the evaluation metrics
    nb_evaluation_metrics.to_csv(BLD / "python" / "evaluation_metrics_nb.csv", index_label="Model")

    # Save the confusion matrix in latex format
    with open(BLD / "python" / "confusion_matrix_nb.tex", "w") as f:
        f.write(conf_matrix_nb.style.to_latex())

    return None


@pytask.mark.depends_on(SRC / "bld" / "python" / "data" / "preprocessed_data.pkl")
@pytask.mark.produces([
    BLD / "python" / "models" / "svm_model.pkl",
    BLD / "python" / "evaluation_metrics_svm.csv",
    BLD / "python" / "confusion_matrix_svm.tex",
])
def task_run_svm(depends_on, produces):
    # Load preprocessed data
    with open(depends_on, "rb") as f:
        data = pickle.load(f)

    # Vectorize the 'Text' column
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Text'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, data['Sentiment'], random_state=42)

    # Fit SVM model
    svm_model, svm_evaluation_metrics, conf_matrix_svm = fit_svm(X_train, y_train, X_test, y_test)

    # Save the model
    with open(produces[0], "wb") as f:
        pickle.dump(svm_model, f)

    # Save the evaluation metrics
    svm_evaluation_metrics.to_csv(produces[1], index_label="Model")

    # Save the confusion matrix in LaTeX format
    conf_matrix_svm.style.to_latex(produces[2])

    return None

import pytask
from pathlib import Path
import pandas as pd
from sentiment_analysis_epp.models.plots import plot_sentiment_hist

SRC = Path(__file__).resolve().parent
BLD = SRC / "bld" / "python"

@pytask.mark.depends_on([
    BLD / "data" / "preprocessed_data.pkl",
    BLD / "python" / "evaluation_metrics_logit.csv",
    BLD / "python" / "evaluation_metrics_nb.csv",
    BLD / "python" / "evaluation_metrics_svm.csv",
])
@pytask.mark.produces(BLD / "python" / "evaluation_metrics.csv")
def task_evaluation_metrics(depends_on, produces):
    # Read in the evaluation metrics for each model
    logit_metrics = pd.read_csv(depends_on[1], index_col="Model")
    nb_metrics = pd.read_csv(depends_on[2], index_col="Model")
    svm_metrics = pd.read_csv(depends_on[3], index_col="Model")

    # Concatenate the evaluation metrics DataFrames
    combined_metrics = pd.concat([logit_metrics, nb_metrics, svm_metrics])

    # Save the combined evaluation metrics as a CSV file
    combined_metrics.to_csv(produces, index_label="Model")