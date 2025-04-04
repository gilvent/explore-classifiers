import numpy as np
from classifiers.naive_bayes import NaiveBayes
from evaluation.accuracy import print_accuracy
from evaluation.confusion_matrix import confusion_matrix, print_confusion_matrix
from utils.data_preprocess import train_test_split
from utils.enums import FeatureType


def main():
    dataset = np.loadtxt(fname="data/wifi_localization.txt")
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]
    feature_types = [
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
    ]
    classes = np.unique(Y)

    train_X, train_Y, test_X, test_Y = train_test_split(
        X=X, Y=Y, unique_classes=classes
    )
    classifier = NaiveBayes()

    trainStatus = classifier.train(
        train_X=train_X, train_Y=train_Y, feature_types=feature_types
    )

    if trainStatus == False:
        return

    pred_Y = classifier.test(test_X)

    print_accuracy(actual_Y=test_Y, pred_Y=pred_Y)

    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    print_confusion_matrix(conf_matrix=conf_matrix, classes=classes)


if __name__ == "__main__":
    main()
