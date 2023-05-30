"""
Model.py Module INFO:

METHODS:
1. NeuralModel: Keras Sequential Neural Network framework.

INPUTS: The following are the inputs for the class and method:
    Class:
        x_test: input test set.
        X_train_res: resampled version of the input training set.
        y_train_res: resampled version fo the output training set.

    NeuralModel Method:
        model: STRING
        layers: INT value
        neurons: INT value
        dropout_val: FLOAT value
        act_func: STRING
        learning_rate: FLOAT value

OUTPUTS:
    NeuralModel Method:
        y_pred_1d: the predictions made from using the model with x_test.
        history: This object keeps all loss values and other metric values in memory
        so that they can be used in the future. The metrics it includes are
        'val_loss', 'val_accuracy', 'loss', 'accuracy'.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from keras.wrappers.scikit_learn import KerasClassifier
import keras as kr
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
import google.cloud.bigquery as gbq
import pandas as pd
from keras.callbacks import EarlyStopping
import time
import google.cloud.bigquery as gbq
import pandas as pd
from GBQ_descriptions import DescriptionsBQ


class Model:
    def __init__(self, x_test, X_train_res, y_train_res):
        self.x_test = x_test
        self.X_train_res = X_train_res
        self.y_train_res = y_train_res

    def FeatureImportance(self, index, rf_import, n_display):
        # combine feature importance with index array
        importances = pd.DataFrame.from_dict(index, orient="index", columns=["Code"])
        importances = pd.concat([pd.DataFrame({"Code": "NA"}, index=[0]), importances])
        importances = importances.head(len(rf_import))
        importances["Importance"] = rf_import

        # sort and cut length to how many we want to display
        importances_sorted = importances.sort_values("Importance", ascending=False)
        importances_sorted = importances_sorted.head(n_display)

        codes_str_list = importances_sorted["Code"].values.tolist()
        to_int_map = map(int, codes_str_list)
        codes = tuple(to_int_map)

        # Created script to utilize GBQ and create a pandas data frame of the descriptions.
        descriptions = DescriptionsBQ(codes)

        return importances_sorted, descriptions

    def PickModel(self, ModelType):

        if ModelType == "Neural Net":
            print("RUNNING NEURAL NET...")
            t0 = time.time()

            def create_model(
                layers=2,
                neurons=100,
                dropout_val=0.2,
                act_func="elu",
                learning_rate=0.001,
            ):

                # Initialize the constructor
                model = Sequential()

                # Add an input layer
                model.add(Dense(neurons, input_shape=(self.X_train_res.shape[1],)))
                model.add(Activation(act_func))
                model.add(Dropout(dropout_val))

                # Hidden Layers
                for layer in range(layers):
                    model.add(Dense(neurons))
                    model.add(Activation(act_func))
                    model.add(Dropout(dropout_val))

                # Add an output layer
                model.add(Dense(2, activation="softmax"))

                # Optimizer Set Up
                adam = kr.optimizers.Adamax(
                    learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
                )
                # Compile model
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer=adam,
                    metrics=["accuracy"],
                )
                return model

            earlystop = EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=1,
                verbose=1,
                mode="auto",
                restore_best_weights=True,
            )
            callbacks_list = [earlystop]

            model = KerasClassifier(build_fn=create_model, verbose=1)

            param_grid = {
                "batch_size": [25],
                "epochs": [50],
                "layers": [5, 10, 20],
                "neurons": [50, 100],
                "dropout_val": [0.2],
                "act_func": ["elu"],
                "learning_rate": [0.001],
            }

            keras_grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring="accuracy"
            )
            self.history = keras_grid_search.fit(
                self.X_train_res,
                self.y_train_res,
                callbacks=callbacks_list,
                validation_split=0.1,
                verbose=1,
            )
            self.best_params = self.history.best_params_

            y_probs = self.history.predict_proba(
                self.x_test
            )  # 2 columns with both probabities.
            probs_calibration = y_probs[:, 1]
            y_pred = []

            for i in range(0, len(y_probs)):
                probs = y_probs[i]
                predicted_index = np.argmax(probs)
                y_pred.append(predicted_index)

            t1 = time.time()
            self.time_elapsed = t1 - t0

            return y_pred, probs_calibration

        elif ModelType == "Random Forest":
            print("RUNNING RANDOM FOREST...")
            t0 = time.time()

            # Real paramater grid to search through.
            param_grid = {
                "n_estimators": [10, 100],
                "max_depth": [10, 50, 100],
                "min_samples_leaf": [3, 5],
            }

            forest_clf = RandomForestClassifier(n_jobs=-1, random_state=0, verbose=1)
            forest_grid_search = GridSearchCV(
                forest_clf, param_grid, cv=10, scoring="accuracy"
            )

            forest_grid_search.fit(self.X_train_res, self.y_train_res.ravel())
            self.best_params = forest_grid_search.best_params_
            self.rf_import = forest_grid_search.best_estimator_.feature_importances_
            y_pred = forest_grid_search.predict(self.x_test)
            probs_calibration = y_pred
            t1 = time.time()
            self.time_elapsed = t1 - t0

            return y_pred, probs_calibration

        elif ModelType == "Naive Bayes":
            print("RUNNING NAIVE BAYES...")
            self.best_params = "No parameters"
            t0 = time.time()

            clf = GaussianNB()

            clf.fit(self.X_train_res, self.y_train_res.ravel())
            y_pred = clf.predict(self.x_test)

            probs_calibration = y_pred
            t1 = time.time()
            self.time_elapsed = t1 - t0

            return y_pred, probs_calibration

        elif ModelType == "Logistic Regression":

            print("RUNNING LOGISTIC REGRESSION...")
            t0 = time.time()

            param_grid = {"C": [1.0], "solver": ["saga"]}

            logitclf = LogisticRegression(multi_class="ovr", n_jobs=-1,)
            logit_grid_search = GridSearchCV(
                logitclf, param_grid, cv=3, scoring="accuracy"
            )

            logit_grid_search.fit(self.X_train_res, self.y_train_res.ravel())
            self.best_params = logit_grid_search.best_params_
            #### Add in logitstic reg. coeficients ----> pull to main script ######

            self.coef = logit_grid_search.best_estimator_.coef_

            ####
            y_pred = logit_grid_search.predict(self.x_test)
            probs_calibration = y_pred

            t1 = time.time()
            self.time_elapsed = t1 - t0

            return y_pred, probs_calibration

        elif ModelType == "Gradient Boosted":
            print("RUNNING GRADIENT BOOST...")
            t0 = time.time()

            param_grid = {
                "max_depth": [3, 4, 5],
                "n_estimators": [25, 50, 100, 200],
                "learning_rate": [0.01, 0.1],
            }

            gbmclf = XGBClassifier(nthread=None)
            gbm_grid_search = GridSearchCV(
                gbmclf, param_grid, cv=3, n_jobs=-1, scoring="accuracy"
            )

            gbm_grid_search.fit(self.X_train_res, self.y_train_res.ravel())
            self.best_params = gbm_grid_search.best_params_
            y_pred = gbm_grid_search.predict(self.x_test)
            probs_calibration = y_pred

            t1 = time.time()
            self.time_elapsed = t1 - t0

            return (
                y_pred,
                probs_calibration,
            )

        else:
            print("There seems to be an issue with selecting a model phase.")
