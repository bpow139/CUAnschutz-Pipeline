"""
Data.py Module INFO:

METHODS:
1. load_data: loads in the data set by either straight from Google Big Query or
    by manually through a directory.

2. process: splitting into training and testing sets and also into labels(outputs)
   and codes(inputs).

INPUTS: Within the class we call in the following.
    path: directory path to the data set. STRING
    vocab_size: how many different codes are there in the data set? INTEGER

The data set that is loaded should have the following columns for processing.
    person_id: unique code corresponding to each patient.
    label: either a 0 or a 1 indicating if patient had the disease.
    diag_med: codes which are used to document different conditions of patient.

OUTPUTS: The following will be the output after running both methods.
    x_train: input variable (diag_med codes) used for training.
    x_test: input variable (diag_med codes) used for testing.
    y_train: output variable (labels) used for training.
    y_test: output variable (labels) used for testing.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing


class Data:
    def __init__(self, data_source_path):
        self.data_source_path = data_source_path
        self.vocab_size = 200

    def load_data(self):
        # Loading in the data from BigQuery or from file source. Needs a lot of work...
        if self.data_source_path is None:
            try:
                data = requests.execute(
                    output_options=bq.QueryOutput.dataframe()
                ).result()
                print("Loaded data successfully from Google Big Query")
                return data

            except:
                print("LOADING ERROR: Unable to load data from Google Big Query")

        # Load data through a path source.
        elif type(self.data_source_path) == str:
            try:
                data = pd.read_csv(self.data_source_path)
                print("succesfully loaded data from " + self.data_source_path)
                return data
            except:
                print("LOADING ERROR: Unable to load data from source path")

        else:
            print(
                "LOADING ERROR: Unable to load data from Google Big Query or a source path"
            )

    def process(self, data):
        # Setting the column 'diag_med' as a string

        data["diag_med"] = data["diag_med"].astype(str)

        # This is splitting the data into 80% TRAIN and 20% TEST. Also the random seed is set to 42.
        train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

        # Split into inputs and labels.
        train_icmed = train_set["diag_med"]
        train_label = train_set["label"]

        test_icmed = test_set["diag_med"]
        test_label = test_set["label"]

        # Tokenizing icmed codes.
        tokenize = Tokenizer(
            num_words=self.vocab_size,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=",",
            char_level=False,
        )

        tokenize.fit_on_texts(train_icmed)
        x_train = tokenize.texts_to_matrix(train_icmed)
        x_test = tokenize.texts_to_matrix(test_icmed)

        # List of tokenized icmed codes
        index = tokenize.index_word

        # Encoding with Sklearn
        encoder = preprocessing.LabelBinarizer()
        encoder.fit(train_label)
        y_train = np.squeeze(encoder.transform(train_label))
        y_test = np.squeeze(encoder.transform(test_label))

        return x_train, x_test, y_train, y_test, index
