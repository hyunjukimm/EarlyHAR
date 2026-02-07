import numpy as np
from torch.utils import data
import torch
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelBinarizer

def computeAUC(predictions, labels):
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    elif labels.shape[1] == 1:
        labels = np.eye(2, dtype='uint8')[labels].squeeze()
    else:
        pass
    label_binarizer = LabelBinarizer().fit(labels)
    y_onehot_test = label_binarizer.transform(labels)

    print(np.array(y_onehot_test).shape, "one hot")
    print(np.array(predictions).shape, "predictions")
    return roc_auc_score(y_onehot_test, predictions,  multi_class= 'ovo', average="macro")

def computeF1(predictions, labels):
    return f1_score(labels, predictions, average='macro')


def computeACC(predictions, labels):
    """
    Computes the accuracy.

    :param predictions: a sequence of prediction tuples of the form (timestamp, class)
    :param labels: a sequence of ground truth labels
    :return: the percentage of correctly classified instances
    """

    # If no predictions or ground truth is provided,
    # the accuracy can not be defined.
    # if not labels or not predictions:
    #     return None

    correct = sum([1 for (prediction, y) in zip(predictions, labels) if prediction == y])
    return correct / len(labels)

def earliness(predictions, ts_length):
    """
    Computes the earliness.

    :param predictions: a sequence of prediction tuples of the form (timestamp, class)
    :param ts_length: the length of the time-series
    :return: the mean timestamp in which predictions are made
    """

    # If the time-series length is zero or no predictions are provided,
    # the earliness can not be defined.
    if ts_length == 0 or not predictions.any():
        return None
    mean_earl = np.mean([number / ts_length for number in predictions])    
    return mean_earl

def earliness_2(predictions):
    """
    Computes the earliness.

    :param predictions: a sequence of prediction tuples of the form (timestamp, class)
    :param ts_length: the length of the time-series
    :return: the mean timestamp in which predictions are made
    """

    # If the time-series length is zero or no predictions are provided,
    # the earliness can not be defined.
    # if not predictions.any():
    #     return None
    mean_earl = np.mean([number for number in predictions])    
    return mean_earl

def harmonic_mean(acc: float, earl: float):
    """
    Computes the harmonic mean as illustrated by Patrick Schäfer et al. 2020
    "TEASER: early and accurate time series classification"

    :param acc: The accuracy of the prediction
    :param earl: The earliness of the prediction
    """
    return (2 * (1 - earl) * acc) / ((1 - earl) + acc)


def exponentialDecay(N):
    tau = 1
    tmax = 7
    t = np.linspace(0, tmax, N)
    y = torch.tensor(np.exp(-t/tau), dtype=torch.float)
    return y

def createNet(n_inputs, n_outputs, n_layers=0, n_units=100, nonlinear=torch.nn.Tanh):
    if n_layers == 0:
        return torch.nn.Linear(n_inputs, n_outputs)
    layers = [torch.nn.Linear(n_inputs, n_units)]
    for i in range(n_layers-1):
        layers.append(nonlinear())
        layers.append(torch.nn.Linear(n_units, n_units))
        layers.append(torch.nn.Dropout(p=0.5))

    layers.append(nonlinear())
    layers.append(torch.nn.Linear(n_units, n_outputs))
    return torch.nn.Sequential(*layers)