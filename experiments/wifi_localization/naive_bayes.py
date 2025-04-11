import numpy as np
from classifiers.naive_bayes import NaiveBayes
from evaluation.metrics import accuracy_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import train_test_split, shuffle_train_test_split
from utils.enums import FeatureType
from utils.display_helpers import to_accuracy_text


def main():
    dataset = np.loadtxt(fname="data/wifi_localization.txt")
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]

    # Initialize the classifier
    classes = np.unique(Y)
    feature_types = [
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
        FeatureType.NUMERICAL.name,
    ]
    classifier = NaiveBayes(unique_classes=classes, feature_types=feature_types)

    # Fit train data to the model
    train_X, train_Y, test_X, test_Y = shuffle_train_test_split(
        X=X, Y=Y, test_split_ratio=0.8
    )
    classifier.train(train_X=train_X, train_Y=train_Y)

    pred_Y = classifier.test(test_X)

    # Display Confusion Matrix
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    accuracy = accuracy_score(actual_Y=test_Y, pred_Y=pred_Y)

    display_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=classes,
        title="Wifi Localization/Naive Bayes (0.8 test split ratio)",
        info=to_accuracy_text(accuracy=accuracy),
    )


if __name__ == "__main__":
    main()
