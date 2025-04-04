import numpy as np
from classifiers.naive_bayes import NaiveBayes
from evaluation.accuracy import print_accuracy
from evaluation.confusion_matrix import confusion_matrix, print_confusion_matrix

def train_test_split(X, Y):
    classes = [1, 2, 3, 4]

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    for c in classes:
        class_inputs = X[np.where(Y == c)]
        class_outputs = Y[np.where(Y == c)]
        test_count_for_class = int(0.3 * len(class_inputs))
        split_index = len(class_inputs) - test_count_for_class

        train_X.extend(class_inputs[:split_index])
        test_X.extend(class_inputs[split_index:])
        train_Y.extend(class_outputs[:split_index])
        test_Y.extend(class_outputs[split_index:])

    return (np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y))


def main():
    dataset = np.loadtxt(fname="data/wifi_localization.txt")
    train_X, train_Y, test_X, test_Y = train_test_split(
        dataset[:, 0:-1], dataset[:, -1]
    )
    classifier = NaiveBayes()

    classifier.train(train_X=train_X, train_Y=train_Y)

    pred_Y = classifier.test(test_X)

    print_accuracy(actual_Y=test_Y, pred_Y=pred_Y)

    classes = np.unique(test_Y)
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    print_confusion_matrix(conf_matrix=conf_matrix, classes=classes)


if __name__ == "__main__":
    main()
