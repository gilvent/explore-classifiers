import numpy as np

def accuracy_score(actual_Y, pred_Y):
    return np.sum(actual_Y == pred_Y, axis=0) / len(actual_Y)
