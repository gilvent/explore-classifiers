import numpy as np
from classifiers.naive_bayes import NaiveBayes
from evaluation.accuracy import accuracy_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import train_test_split, shuffle_train_test_split
from utils.enums import FeatureType
from utils.display_helpers import to_accuracy_text


def main():
    dataset = np.genfromtxt(
        fname="data/winequality_white.csv", delimiter=";", dtype=str
    )
    # Remove header row
    dataset = dataset[1:]
    dataset = dataset.astype(float)
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]

    # Initialize the classifier
    classes = np.unique(Y)
    feature_types = [
        FeatureType.NUMERICAL.name,  # fixed acidity
        FeatureType.NUMERICAL.name,  # volatile acidity
        FeatureType.NUMERICAL.name,  # citric acid
        FeatureType.NUMERICAL.name,  # residual sugar
        FeatureType.NUMERICAL.name,  # chlorides
        FeatureType.NUMERICAL.name,  # free sulfur dioxide
        FeatureType.NUMERICAL.name,  # total sulfur dioxide
        FeatureType.NUMERICAL.name,  # density
        FeatureType.NUMERICAL.name,  # pH
        FeatureType.NUMERICAL.name,  # sulphates
        FeatureType.NUMERICAL.name,  # alcohol
    ]
    classifier = NaiveBayes(unique_classes=classes, feature_types=feature_types)

    # Fit training data to the model
    train_X, train_Y, test_X, test_Y = shuffle_train_test_split(
        X=X, Y=Y, test_split_ratio=0.3
    )
    classifier.train(train_X=train_X, train_Y=train_Y)

    pred_Y = classifier.test(test_X)

    # Display Confusion Matrix
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    accuracy = accuracy_score(actual_Y=test_Y, pred_Y=pred_Y)

    display_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=classes,
        title="White Wine Quality/Naive Bayes (0.3 test split ratio)",
        info=to_accuracy_text(accuracy=accuracy),
    )


if __name__ == "__main__":
    main()
