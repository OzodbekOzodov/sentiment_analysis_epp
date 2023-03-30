import sys
import os
import pytask
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
# from config import SRC
SRC = Path(__file__).parent.resolve()
#BLD = SRC.joinpath("..", "..", "bld").resolve()
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


# DATA_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data_management", "data.csv")



"""
@pytask.mark.depends_on(DATA_PATH)
@pytask.mark.produces(["logit_model", "naive_bayes_model", "svm_model", "evaluation_metrics"])
def task_run_models(depends_on, produces):
    # Read the data
    data = pd.read_csv(depends_on, encoding='ISO-8859-1')

    # Preprocess the 'bag_of_words' column
    vectorizer = CountVectorizer()
    data['bag_of_words'] = vectorizer.fit_transform(data['Text'])

    # Run the models
    logit_model, logit_evaluation = fit_logit_model(data)
    nb_model, nb_evaluation = fit_naive_bayes(data)
    svm_model, svm_evaluation = fit_svm(data)

    # Save the evaluation metrics
    evaluation_metrics = pd.concat([logit_evaluation, nb_evaluation, svm_evaluation], axis=0)
    evaluation_metrics.index = ["logit", "naive_bayes", "svm"]
    evaluation_metrics.to_csv(produces[3])

    # Store the models in memory
    produces[0] = logit_model
    produces[1] = nb_model
    produces[2] = svm_model

    """
