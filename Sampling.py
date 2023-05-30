"""
Sampling.py Module INFO:

METHODS:
1. PickSampleMethod: Selects the sample technique that will be inputted.

INPUTS:
x_train: Training set for inputs.
y_train: Training set for output.
Sample: STRING from the main script which selects the type of sampling.

OUTPUTS:
X_train_res: Resampled version of the training input set.
y_train_res: Resampled version of the training output set.
Sample: Sample technique used.
"""

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler


class Sampling:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def PickSampleMethod(self, Sample):
        print(
            "Before OverSampling, counts of label '1': {}".format(
                sum(self.y_train == 1)
            )
        )
        print(
            "Before OverSampling, counts of label '0': {} \n".format(
                sum(self.y_train == 0)
            )
        )

        if Sample == "RandomOverSampler":
            sm = RandomOverSampler(random_state=42)
        elif Sample == "SMOTE":
            sm = SMOTE()
        elif Sample == "SMOTEENN":
            sm = SMOTEENN()
        elif Sample == "SMOTETomek":
            sm = SMOTETomek()
        elif Sample == "ADASYN":
            sm = ADASYN()
        elif Sample == "ClusterCentroids":
            sm = ClusterCentroids()
        elif Sample == "RandomUnderSampler":
            sm = RandomUnderSampler()
        else:
            print("Incorrect Sampling Method has been input.")

        print("The sampling method being used is {}".format(sm))

        X_train_res, y_train_res = sm.fit_resample(self.x_train, self.y_train)

        if Sample == "RandomOverSampler":
            print(
                "After OverSampling, count of label '1': {}".format(
                    sum(y_train_res == 1)
                )
            )
            print(
                "After OverSampling, counts of label '0': {}".format(
                    sum(y_train_res == 0)
                )
            )
        elif Sample == "SMOTE":
            pass
        elif Sample == "SMOTEENN":
            pass
        elif Sample == "SMOTETomek":
            pass
        elif Sample == "ADASYN":
            pass
        elif Sample == "ClusterCentroids":
            pass
        elif Sample == "RandomUnderSampler":
            pass
        else:
            print("No sampling has been done.")

        return X_train_res, y_train_res, Sample
