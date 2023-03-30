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


import pandas as pd
import pickle
from pathlib import Path
from sentiment_analysis_epp.models.models import fit_logit_model, fit_naive_bayes, fit_svm, fit_crf
import pytask
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

@pytask.mark.depends_on(SRC / "bld" / "python" / "data" / "preprocessed_data.pkl")
@pytask.mark.produces([
    BLD / "python" / "models" / "logit_model.pkl",
    BLD / "python" / "models" / "naive_bayes_model.pkl",
    BLD / "python" / "models" / "svm_model.pkl",
    BLD / "python" / "models" / "crf_model.pkl",
    BLD / "python" / "evaluation_metrics.csv",
])
def task_run_models(depends_on, produces):
    # Load preprocessed data
    with open(depends_on, "rb") as f:
        data = pickle.load(f)

    # Vectorize the 'Text' column
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Text'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, data['Sentiment'], random_state=42)

    # Fit models
    logit_model, logit_evaluation = fit_logit_model(X_train, y_train, X_test, y_test)
    nb_model, nb_evaluation = fit_naive_bayes(X_train, y_train, X_test, y_test)
    svm_model, svm_evaluation = fit_svm(X_train, y_train, X_test, y_test)
    crf_model, crf_evaluation = fit_crf(X_train, y_train, X_test, y_test)
    # Save models
    with open(produces[0], "wb") as f:
        pickle.dump(logit_model, f)
    with open(produces[1], "wb") as f:
        pickle.dump(nb_model, f)
    with open(produces[2], "wb") as f:
        pickle.dump(svm_model, f)
    with open(produces[3], "wb") as f:
        pickle.dump(crf_model, f)


    # Save evaluation metrics
    evaluation_metrics = pd.concat([logit_evaluation, nb_evaluation, svm_evaluation, crf_evaluation], axis=0)
    evaluation_metrics.index = ["logit", "naive_bayes", "svm", "crf"]
    evaluation_metrics.to_csv(produces[3], index_label="Model")

    return None



from sentiment_analysis_epp.models.models import lda_topic_modeling
@pytask.mark.depends_on(SRC / "bld" / "python" / "data" / "preprocessed_data.pkl")
@pytask.mark.produces([
    BLD / "python" / "models" / "data_with_topics.csv",
    BLD / "python" / "models" / "topic_examples.csv",
])
def task_run_topic_lda(depends_on, produces):
    # Load preprocessed data
    with open(depends_on, "rb") as f:
        data = pickle.load(f)

    # Run LDA topic modeling
    data_with_topics, topic_examples = lda_topic_modeling(data, n_topics=20, n_keywords=20) #, random_state=42)

    # Save DataFrames
    data_with_topics.to_csv(produces[0],index = False)
    topic_examples.to_csv(produces[1], index=False)


    return None
