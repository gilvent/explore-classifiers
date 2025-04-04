import numpy as np

def print_accuracy(true_Y, pred_Y):
    accuracy = np.sum(true_Y == pred_Y, axis=0) / len(true_Y)

    print("Accuracy of predictions:", accuracy)
    return accuracy