"""
Main.py INFO:

This is the main script that will be run. It calls upon several other scripts/
classes to perform sub-routines. They are the following

1. Data.py: Load in data, split and tokenize.
2. Sampling.py: Selecting a sampling technique.
3. Model.py: Selecting a model and then training data with it.
4. Output.py: Graphical displays of metrics used to score different techniques.
5. Save.py: Saving results in an orderly fashion.
"""

import os
from datetime import datetime
from Data import Data
from Model import Model
from Output import Output
from Sampling import Sampling
import pandas as pd
import numpy as np

"""
Types of models:
"Neural Net",
"Logistic Regression",
"Random Forest",
"Naive Bayes",
"Gradient Boosted"
"""
# Pick a sampling methods and model.
SamplingMethodString = "SMOTE"
ModelType = ["Random Forest", "Logistic Regression"]
FeatureImportance = True

# Assign data acquisition directory.
data_source_path = (
    "/Users/brettpowers/Work/compassMachineLearning/Data/hf_new_small.csv"
)

# Create new directory for current run.
Results_dir = "/Users/brettpowers/Work/compassMachineLearning/Results"
date_base = datetime.now()
date = str(date_base.strftime("%d-%m-%Y-%H%M%S"))
Save_path = os.path.join(Results_dir, date)
os.makedirs(Save_path)

# Load and transform data.
data_object = Data(data_source_path)
data = data_object.load_data()
x_train, x_test, y_train, y_test, index = data_object.process(data)

# Resample data.
SampleSet = Sampling(x_train, y_train)
X_train_res, y_train_res, Sample = SampleSet.PickSampleMethod(SamplingMethodString)

# Create a text file for a simple report of run.
filename = "Report_{}".format(date)
report_path = os.path.join(Save_path, filename)
report = open(report_path, "w")
report.write(
    "ANALYSIS REPORT \n {} \n {}".format(
        date_base.strftime("%m/%d/%y"), date_base.strftime("%H:%M")
    )
)
report.close()

# Counter to indicate when feature importance analysis is ready to be done.
heat_map_counter = 0
# For-Loop through model types
for model in ModelType:
    # Set model and train
    Model_create = Model(x_test, X_train_res, y_train_res)
    y_pred, probs_calibration = Model_create.PickModel(model)

    # Run feature importance analysis.
    if FeatureImportance:

        if model == "Random Forest":
            rf_import = Model_create.rf_import
            importances_sorted, descriptions = Model_create.FeatureImportance(
                index, rf_import, n_display=25
            )
            importances_sorted = importances_sorted.reset_index()
            feature_importance_df = importances_sorted.join(descriptions)

            heat_map_counter += 1

        if model == "Logistic Regression":
            coef = Model_create.coef
            coef = coef.transpose()

            coefficients = pd.DataFrame.from_dict(
                index, orient="index", columns=["Code"]
            )
            coefficients = pd.concat(
                [pd.DataFrame({"Code": "NA"}, index=[0]), coefficients]
            )
            coefficients = coefficients.head(len(coef))
            coefficients["Coefficient"] = coef

            heat_map_counter += 1

    # # Output important results
    Output_create = Output(
        y_test, y_pred, model, probs_calibration, save_path=Save_path
    )

    f1, auc = Output_create.Scores()
    Output_create.plot_confusion_matrix()
    Output_create.plot_ROC()
    Output_create.plot_CalibrationCurve()

    if heat_map_counter == 2:
        Output_create.plot_Heat(coefficients, feature_importance_df, n_display=10)

        print(feature_importance_df)

        print(coefficients)

    with open(report_path, "a") as report:

        L = [
            "---------------------------------------------------------------- \n",
            "---------------------------------------------------------------- \n",
            "Model Type: {} \n".format(model),
            "Compute Time: {} \n".format(Model_create.time_elapsed),
            "F1 Score: {} \n".format(f1),
            "AUC: {} \n".format(auc),
            "Parameters Used: {} \n ".format(Model_create.best_params),
            "---------------------------------------------------------------- \n",
            "----------------------------------------------------------------",
        ]

        report.writelines(L)
