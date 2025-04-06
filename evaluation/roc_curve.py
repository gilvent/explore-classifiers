import matplotlib.pyplot as plt
import numpy as np

def predictions_by_threshold(probabilities: np.ndarray, threshold: float = 0.1):
    result = []
    for p in probabilities:
        result.append(1 if p >= threshold else 0)
    
    return result

def print_roc_curve(fpr: np.ndarray, tpr: np.ndarray):
    # TODO compute AUC properly
    roc_auc = 0
    
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()