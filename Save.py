"""
Save.py Module INFO:
This module will produce a PDF report of the basic information about the run.

METHODS:
No Methods in this class.

INPUTS:
f1
auc
CM_fig
ROC_fig
vocab_size
layers
neurons,
dropout_val
act_func
learning_rate
SamplingMethodString
save_path

OUTPUTS:
No Outputs. Will save a PDF file into directory path used.
"""

# from reportlab.lib.utils import ImageReader
# from reportlab.pdfgen import canvas
import os
from io import BytesIO


class Save:
    def __init__(
        self,
        f1,
        auc,
        CM_fig,
        ROC_fig,
        vocab_size,
        layers,
        neurons,
        dropout_val,
        act_func,
        learning_rate,
        SamplingMethodString,
        save_path,
    ):
        self.f1 = f1
        self.auc = auc
        self.save_path = save_path
        self.vocab_size = vocab_size
        self.layers = layers
        self.neurons = neurons
        self.dropout_val = dropout_val
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.SamplingMethodString = SamplingMethodString

        # report = open('')

        # c = canvas.Canvas(os.path.join(self.save_path, "ScoreReport.pdf"))
        # c.setLineWidth(1.5)
        # c.setFont('Helvetica', 12)
        #
        # # Model and Sampling.
        # c.drawString(30, 800, "Sampling Method: {}".format(self.SamplingMethodString))
        #
        # # Stating parameters used.
        # c.drawString(30, 775, 'Parameters: VOCAB SIZE: {} -- LAYERS: {} -- NEURONS: {} -- DROPOUT
        # VALUE: {}'.format(self.vocab_size, self.layers, self.neurons, self.dropout_val))
        # c.drawString(100, 750, 'ACTIVATION FUNCTION: {} -- LEARNING RATE: {}'.format(self.act_func, self.learning_rate))
        #
        # # Saving the scores
        # c.drawString(30, 725, 'F-1 Score: {}'.format(self.f1))
        # c.drawString(30, 700, 'AUC: {}'.format(self.auc))
        #
        # # Saving the figures
        # imgdata = BytesIO()
        # CM_fig.savefig(imgdata, format='png')
        # imgdata.seek(0)
        # image = ImageReader(imgdata)
        # c.drawImage(image, 100, 450, 300, 200)
        #
        # imgdata = BytesIO()
        # ROC_fig.savefig(imgdata, format='png')
        # imgdata.seek(0)
        # image = ImageReader(imgdata)
        # c.drawImage(image, 100, 250, 300, 200)
        #
        # c.save()
