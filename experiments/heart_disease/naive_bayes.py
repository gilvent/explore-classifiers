import numpy as np
from classifiers.naive_bayes import NaiveBayes
from evaluation.accuracy import print_accuracy
from evaluation.confusion_matrix import confusion_matrix, print_confusion_matrix
from utils.data_preprocess import train_test_split
from utils.enums import FeatureType
from evaluation.roc_curve import print_roc_curve, predictions_by_threshold


def main():
    dataset = np.loadtxt(
        fname="data/heart_disease_cleveland_processed.txt", delimiter=",", dtype=str
    )

    # Remove rows with missing values
    filtered_dataset = dataset[~np.char.equal(dataset, "?").any(axis=1)]
    filtered_dataset = filtered_dataset.astype(float)
    X = filtered_dataset[:, 0:-1]
    Y = filtered_dataset[:, -1]

    # Since all target > 0 is classified as having heart disease, we convert the value into 1
    Y[Y > 0] = 1

    feature_types = [
        FeatureType.NUMERICAL.name,  # age
        FeatureType.CATEGORICAL.name,  # sex
        FeatureType.CATEGORICAL.name,  # cp
        FeatureType.NUMERICAL.name,  # trestbps
        FeatureType.NUMERICAL.name,  # chol
        FeatureType.CATEGORICAL.name,  # fbs
        FeatureType.CATEGORICAL.name,  # restecg
        FeatureType.NUMERICAL.name,  # thalach
        FeatureType.CATEGORICAL.name,  # exang
        FeatureType.NUMERICAL.name,  # oldpeak
        FeatureType.CATEGORICAL.name,  # slope
        FeatureType.NUMERICAL.name,  # ca
        FeatureType.CATEGORICAL.name,  # thal
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

    # Prediction accuracy on test data
    print_accuracy(actual_Y=test_Y, pred_Y=pred_Y)

    # Confusion matrix on test data
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    print_confusion_matrix(conf_matrix=conf_matrix, classes=classes)


    # ROC Curve
    # We use posterior probabilities for affirmative class (where heart disease classified as PRESENT)
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
