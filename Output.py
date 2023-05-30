"""
Output.py Module INFO:

METHODS:
1. Scores: These are some simple scoring features.
2. plot_confusion_matrix: Plotting a confusion matrix after testing.
3. plot_ROC: Plotting a ROC.

INPUTS:
y_test: Testing output set.
y_pred_1d: Prediction of outputs from test input set.
save_path: directory path for the saving module.

OUTPUTS:
f1: F1 score
auc: AUC (area under curve metric)
fig1: confustion matrix figure.
fig2: ROC figure.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve


class Output:
    def __init__(self, y_test, y_pred_1d, model, probs_calibration, save_path):
        self.y_test = y_test
        self.y_pred_1d = y_pred_1d
        self.model = model
        self.probs_calibration = probs_calibration
        self.save_path = save_path
        self.cnf_matrix = confusion_matrix(self.y_test, self.y_pred_1d)
        np.set_printoptions(precision=2)
        self.labels = [0, 1]

    def Scores(self):
        f1 = f1_score(self.y_test, self.y_pred_1d, average="binary")
        print("f-score metric in the testing dataset: {}".format(f1))
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, self.y_pred_1d)
        auc = metrics.auc(fpr, tpr)
        print("AUC of test dataset is: {}".format(auc))

        return f1, auc

    def plot_confusion_matrix(self):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = self.cnf_matrix
        title = "Confusion matrix, without normalization"
        classes = self.labels
        normalize = False

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            pass  # print('Confusion matrix, without normalization')

        # print(cm)

        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.autoscale()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(os.path.join(self.save_path, "CM_{}".format(self.model)))
        # plt.show()
        # plt.close()

    def plot_ROC(self):
        fpr, tpr, threshold = metrics.roc_curve(self.y_test, self.y_pred_1d)
        roc_auc = metrics.auc(fpr, tpr)

        plt.figure()
        plt.title("Receiver Operating Characteristic")
        plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.savefig(os.path.join(self.save_path, "ROC_{}".format(self.model)))
        # plt.show()
        # plt.close()

    def plot_CalibrationCurve(self):
        fop, mpv = calibration_curve(
            self.y_test, self.probs_calibration, n_bins=10, normalize=True
        )

        plt.figure()
        plt.title("Calibration Curve")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(mpv, fop, marker=".")
        plt.savefig(os.path.join(self.save_path, "CAL_{}".format(self.model)))
        # plt.show()
        # plt.close()

    def plot_Heat(self, coefficients, feature_importance_df, n_display):
        print(" Heat Map being generated.....")

        heat_df = pd.merge(feature_importance_df, coefficients, how="inner", on="Code")

        conditions = [
            (heat_df["Coefficient"] < 0),
            (heat_df["Coefficient"] == 0),
            (heat_df["Coefficient"] > 0),
        ]
        values = [-1, 0, 1]
        heat_df["Direction"] = np.select(conditions, values)

        heat_df["Heat Val"] = heat_df["Importance"] * heat_df["Direction"]
        heat_df.sort_values("Importance", ascending=False, inplace=True)

        heat_df.drop(
            ["Importance", "Coefficient", "Direction", "Code", "index"],
            axis=1,
            inplace=True,
        )
        heat_df.set_index(0, inplace=True)

        heat_final = heat_df.head(n_display)
        print(heat_final)
        sns.heatmap(heat_final, cmap="PiYG", center=0)

        plt.savefig(os.path.join(self.save_path, "HEAT_MAP"))

        # plt.show()
        # plt.close()
