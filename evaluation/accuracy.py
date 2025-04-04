import numpy as np

def print_accuracy(actual_Y, pred_Y):
    accuracy = np.sum(actual_Y == pred_Y, axis=0) / len(actual_Y)

    print("Accuracy of predictions:", accuracy)
    return accuracy