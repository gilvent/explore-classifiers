import numpy as np
from classifiers.naive_bayes import NaiveBayes
from evaluation.accuracy import accuracy_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import (
    to_seconds_since_midnight,
)
from utils.enums import FeatureType
from utils.display_helpers import to_accuracy_text
from evaluation.roc_curve import print_roc_curve, predictions_by_threshold


def preprocess(dataset):
    # Convert the second column (date time string) into numerical value
    X_2nd_col = dataset[:, 1]
    updated_X_2nd_col = []

    for datestr in X_2nd_col:
        time_in_seconds = to_seconds_since_midnight(datestr.replace('"', ""))
        updated_X_2nd_col.append(time_in_seconds)

    updated_X_2nd_col = np.array(updated_X_2nd_col)

    # Combine the updated second column with the rest, we don't use the first column
    X_rest = dataset[:, 2:-1].astype(float)
    X = np.hstack((updated_X_2nd_col.reshape(-1, 1), X_rest))
    Y = dataset[:, -1].astype(float)

    return (X, Y)


def main():
    training_set = np.loadtxt(
        fname="data/room_occupancy_datatraining.txt", delimiter=",", dtype=str
    )
    test_set = np.loadtxt(fname="data/room_occupancy_datatest.txt", delimiter=",", dtype=str)

    train_X, train_Y = preprocess(training_set)
    test_X, test_Y = preprocess(test_set)

    # Initialize the model
    classes = np.unique(train_Y)
    feature_types = [
        FeatureType.NUMERICAL.name,  # date time converted into numerical values based on time
        FeatureType.NUMERICAL.name,  # temperature
        FeatureType.NUMERICAL.name,  # humidty, continuous
        FeatureType.NUMERICAL.name,  # light
        FeatureType.NUMERICAL.name,  # CO2, continuous
        FeatureType.NUMERICAL.name,  # humidity ratio
    ]
    classifier = NaiveBayes(unique_classes=classes, feature_types=feature_types)

    classifier.train(train_X=train_X, train_Y=train_Y)

    pred_Y = classifier.test(test_X)

    # Confusion Matrix
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    accuracy = accuracy_score(actual_Y=test_Y, pred_Y=pred_Y)

    display_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=classes,
        title="Room Occupation/Naive Bayes",
        info=to_accuracy_text(accuracy=accuracy),
    )

    # ROC Curve
    affirmative_class_posteriors = classifier.posterior_probabilities(
        test_X=test_X, target_class=classes[np.where(classes == 1)][0]
    )
    tpr = []
    fpr = []

    for t in np.arange(0, 1.05, 0.05):
        thresholded_pred = predictions_by_threshold(
            probabilities=affirmative_class_posteriors, threshold=t
        )
        conf_matrix = confusion_matrix(
            classes=classes, actual_Y=test_Y, pred_Y=thresholded_pred
        )
        tp = conf_matrix[1][1]
        fp = conf_matrix[0][1]
        tn = conf_matrix[0][0]
        fn = conf_matrix[1][0]

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    print_roc_curve(fpr, tpr)


if __name__ == "__main__":
    main()
